/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <stddef.h>
#include <string>

#include "syntaxnet/affix.h"
#include "syntaxnet/char_ngram_string_extractor.h"
#include "syntaxnet/feature_extractor.h"
#include "syntaxnet/segmenter_utils.h"
#include "syntaxnet/sentence.pb.h"
#include "syntaxnet/sentence_batch.h"
#include "syntaxnet/term_frequency_map.h"
#include "syntaxnet/utils.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"

// A task that collects term statistics over a corpus and saves a set of
// term maps; these saved mappings are used to map strings to ints in both the
// chunker trainer and the chunker processors.

using tensorflow::DEVICE_CPU;
using tensorflow::DT_INT32;
using tensorflow::DT_STRING;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::errors::InvalidArgument;

namespace syntaxnet {
namespace {

// Helper function to load the TaskSpec either from the `task_context`
// or the `task_context_str` arguments of the op.
void LoadSpec(OpKernelConstruction *context, TaskSpec *task_spec) {
  string file_path, data;
  OP_REQUIRES_OK(context, context->GetAttr("task_context", &file_path));
  if (!file_path.empty()) {
    OP_REQUIRES_OK(context, ReadFileToString(tensorflow::Env::Default(),
                                             file_path, &data));
  } else {
    OP_REQUIRES_OK(context, context->GetAttr("task_context_str", &data));
  }
  OP_REQUIRES(context, TextFormat::ParseFromString(data, task_spec),
              InvalidArgument("Could not parse task context from ", data));
}

class LexiconBuilder : public OpKernel {
 public:
  explicit LexiconBuilder(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("corpus_name", &corpus_name_));
    OP_REQUIRES_OK(context, context->GetAttr("lexicon_max_prefix_length",
                                             &max_prefix_length_));
    OP_REQUIRES_OK(context, context->GetAttr("lexicon_max_suffix_length",
                                             &max_suffix_length_));
    LoadSpec(context, task_context_.mutable_spec());

    int min_length, max_length;
    OP_REQUIRES_OK(context, context->GetAttr("lexicon_min_char_ngram_length",
                                             &min_length));
    OP_REQUIRES_OK(context, context->GetAttr("lexicon_max_char_ngram_length",
                                             &max_length));
    bool add_terminators, mark_boundaries;
    OP_REQUIRES_OK(context,
                   context->GetAttr("lexicon_char_ngram_include_terminators",
                                    &add_terminators));
    OP_REQUIRES_OK(context,
                   context->GetAttr("lexicon_char_ngram_mark_boundaries",
                                    &mark_boundaries));
    char_ngram_string_extractor_.set_min_length(min_length);
    char_ngram_string_extractor_.set_max_length(max_length);
    char_ngram_string_extractor_.set_add_terminators(add_terminators);
    char_ngram_string_extractor_.set_mark_boundaries(mark_boundaries);
    char_ngram_string_extractor_.Setup(task_context_);
  }

  // Counts term frequencies.
  void Compute(OpKernelContext *context) override {
    // Term frequency maps to be populated by the corpus.
    TermFrequencyMap words;
    TermFrequencyMap lcwords;
    TermFrequencyMap tags;
    TermFrequencyMap categories;
    TermFrequencyMap labels;
    TermFrequencyMap chars;
    TermFrequencyMap morphs;

    TermFrequencyMap source_words;
    TermFrequencyMap source_lcwords;
    TermFrequencyMap source_tags;
    TermFrequencyMap source_categories;
    TermFrequencyMap source_labels;
    TermFrequencyMap source_chars;
    TermFrequencyMap source_morphs;

    TermFrequencyMap char_ngrams;

    // Affix tables to be populated by the corpus.
    AffixTable prefixes(AffixTable::PREFIX, max_prefix_length_);
    AffixTable suffixes(AffixTable::SUFFIX, max_suffix_length_);

    // Tag-to-category mapping.
    TagToCategoryMap tag_to_category;


    // Affix tables to be populated by the corpus.
    AffixTable source_prefixes(AffixTable::PREFIX, max_prefix_length_);
    AffixTable source_suffixes(AffixTable::SUFFIX, max_suffix_length_);

    // Tag-to-category mapping.
    TagToCategoryMap source_tag_to_category;

    // Make a pass over the corpus.
    int64 num_tokens = 0;
    int64 num_documents = 0;
    Sentence *document;
    TextReader corpus(*task_context_.GetInput(corpus_name_), &task_context_);
    while ((document = corpus.Read()) != nullptr) {
      // Gather token information.
      for (int t = 0; t < document->target().token_size(); ++t) {
        // Get token and lowercased word.
        auto &token = document->target().token(t);
        string word = token.word();
        utils::NormalizeDigits(&word);
        string lcword = tensorflow::str_util::Lowercase(word);

        // Make sure the token does not contain a newline.
        CHECK(lcword.find('\n') == string::npos);

        // Increment frequencies (only for terms that exist).
        if (!word.empty() && !HasSpaces(word)) words.Increment(word);
        if (!lcword.empty() && !HasSpaces(lcword)) lcwords.Increment(lcword);
        if (!token.tag().empty()) tags.Increment(token.tag());
        if (!token.category().empty()) categories.Increment(token.category());
        if (!token.label().empty()) labels.Increment(token.label());

        if(token.HasExtension(TokenMorphology::morphology)){
            auto& morph = token.GetExtension(TokenMorphology::morphology);
            for(int m=0; m < morph.attribute_size(); m++){
                auto& attr = morph.attribute(m);
                morphs.Increment(attr.name());
            }
        }

        // Add prefixes/suffixes for the current word.
        prefixes.AddAffixesForWord(word.c_str(), word.size());
        suffixes.AddAffixesForWord(word.c_str(), word.size());

        // Add mapping from tag to category.
        tag_to_category.SetCategory(token.tag(), token.category());

        // Add characters.
        std::vector<tensorflow::StringPiece> char_sp;
        SegmenterUtils::GetUTF8Chars(word, &char_sp);
        for (const auto &c : char_sp) {
          const string c_str = c.ToString();
          if (!c_str.empty() && !HasSpaces(c_str)) chars.Increment(c_str);
        }

        // Add character ngrams.
        char_ngram_string_extractor_.Extract(
            word, [&](const string &char_ngram) {
              char_ngrams.Increment(char_ngram);
            });

        // Update the number of processed tokens.
        ++num_tokens;
      }

      // Gather token information.
      for (int t = 0; t < document->source().token_size(); ++t) {
        // Get token and lowercased word.
        auto &token = document->source().token(t);
        string word = token.word();
        utils::NormalizeDigits(&word);
        string lcword = tensorflow::str_util::Lowercase(word);

        // Make sure the token does not contain a newline.
        CHECK(lcword.find('\n') == string::npos);

        // Increment frequencies (only for terms that exist).
        if (!word.empty() && !HasSpaces(word)) source_words.Increment(word);
        if (!lcword.empty() && !HasSpaces(lcword)) source_lcwords.Increment(lcword);
        if (!token.tag().empty()) source_tags.Increment(token.tag());
        if (!token.category().empty()) source_categories.Increment(token.category());
        if (!token.label().empty()) source_labels.Increment(token.label());

        if(token.HasExtension(TokenMorphology::morphology)){
            auto& morph = token.GetExtension(TokenMorphology::morphology);
            for(int m=0; m < morph.attribute_size(); m++){
                auto& attr = morph.attribute(m);
                source_morphs.Increment(attr.name());
            }
        }

        // Add prefixes/suffixes for the current word.
        source_prefixes.AddAffixesForWord(word.c_str(), word.size());
        source_suffixes.AddAffixesForWord(word.c_str(), word.size());

        // Add mapping from tag to category.
        source_tag_to_category.SetCategory(token.tag(), token.category());

        // Add characters.
        std::vector<tensorflow::StringPiece> char_sp;
        SegmenterUtils::GetUTF8Chars(word, &char_sp);
        for (const auto &c : char_sp) {
          const string c_str = c.ToString();
          if (!c_str.empty() && !HasSpaces(c_str)) source_chars.Increment(c_str);
        }

        // Update the number of processed tokens.
        ++num_tokens;
      }

      delete document;
      ++num_documents;
    }
    LOG(INFO) << "Term maps collected over " << num_tokens << " tokens from "
              << num_documents << " documents";

    // Write mappings to disk.
    words.Save(TaskContext::InputFile(*task_context_.GetInput("word-map")));
    lcwords.Save(TaskContext::InputFile(*task_context_.GetInput("lcword-map")));
    tags.Save(TaskContext::InputFile(*task_context_.GetInput("tag-map")));
    categories.Save(
        TaskContext::InputFile(*task_context_.GetInput("category-map")));
    labels.Save(TaskContext::InputFile(*task_context_.GetInput("label-map")));
    chars.Save(TaskContext::InputFile(*task_context_.GetInput("char-map")));
    morphs.Save(TaskContext::InputFile(*task_context_.GetInput("morphology-map")));

    // Optional, for backwards-compatibility with existing specs.
    TaskInput *char_ngrams_input = task_context_.GetInput("char-ngram-map");
    if (char_ngrams_input->part_size() > 0) {
      char_ngrams.Save(TaskContext::InputFile(*char_ngrams_input));
    }

    // Write affixes to disk.
    WriteAffixTable(prefixes, TaskContext::InputFile(
                                  *task_context_.GetInput("prefix-table")));
    WriteAffixTable(suffixes, TaskContext::InputFile(
                                  *task_context_.GetInput("suffix-table")));

    // Write tag-to-category mapping to disk.
    source_tag_to_category.Save(
        TaskContext::InputFile(*task_context_.GetInput("tag-to-category")));

    // Write mappings to disk.
    source_words.Save(TaskContext::InputFile(*task_context_.GetInput("source-word-map")));
    source_lcwords.Save(TaskContext::InputFile(*task_context_.GetInput("source-lcword-map")));
    source_tags.Save(TaskContext::InputFile(*task_context_.GetInput("source-tag-map")));
    source_categories.Save(
        TaskContext::InputFile(*task_context_.GetInput("source-category-map")));
    source_labels.Save(TaskContext::InputFile(*task_context_.GetInput("source-label-map")));
    source_chars.Save(TaskContext::InputFile(*task_context_.GetInput("source-char-map")));
    source_morphs.Save(TaskContext::InputFile(*task_context_.GetInput("source-morphology-map")));

    // Write affixes to disk.
    WriteAffixTable(source_prefixes, TaskContext::InputFile(
                                  *task_context_.GetInput("source-prefix-table")));
    WriteAffixTable(source_suffixes, TaskContext::InputFile(
                                  *task_context_.GetInput("source-suffix-table")));

    // Write tag-to-category mapping to disk.
    source_tag_to_category.Save(
        TaskContext::InputFile(*task_context_.GetInput("source-tag-to-category")));
  }

 private:
  // Returns true if the word contains spaces.
  static bool HasSpaces(const string &word) {
    for (char c : word) {
      if (c == ' ') return true;
    }
    return false;
  }

  // Writes an affix table to a task output.
  static void WriteAffixTable(const AffixTable &affixes,
                              const string &output_file) {
    ProtoRecordWriter writer(output_file);
    affixes.Write(&writer);
  }

  // Name of the context input to compute lexicons.
  string corpus_name_;

  // Max length for prefix table.
  int max_prefix_length_;

  // Max length for suffix table.
  int max_suffix_length_;

  // Extractor for character n-gram strings.
  CharNgramStringExtractor char_ngram_string_extractor_;

  // Task context used to configure this op.
  TaskContext task_context_;
};

REGISTER_KERNEL_BUILDER(Name("LexiconBuilder").Device(DEVICE_CPU),
                        LexiconBuilder);

class FeatureSize : public OpKernel {
 public:
  explicit FeatureSize(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("arg_prefix", &arg_prefix_));
    OP_REQUIRES_OK(context, context->MatchSignature(
                                {}, {DT_INT32, DT_INT32, DT_INT32, DT_INT32}));
    string data;
    OP_REQUIRES_OK(context, ReadFileToString(tensorflow::Env::Default(),
                                             task_context_path, &data));
    OP_REQUIRES(
        context,
        TextFormat::ParseFromString(data, task_context_.mutable_spec()),
        InvalidArgument("Could not parse task context at ", task_context_path));

    //LoadSpec(context, task_context_.mutable_spec());
    string label_map_path =
        TaskContext::InputFile(*task_context_.GetInput("label-map"));
    label_map_ = SharedStoreUtils::GetWithDefaultName<TermFrequencyMap>(
        label_map_path, 0, 0);
    string word_map_path =
        TaskContext::InputFile(*task_context_.GetInput("word-map"));
    word_map_ = SharedStoreUtils::GetWithDefaultName<TermFrequencyMap>(
        word_map_path, 0, 0);
    string tag_map_path =
        TaskContext::InputFile(*task_context_.GetInput("tag-map"));
    tag_map_ = SharedStoreUtils::GetWithDefaultName<TermFrequencyMap>(
        tag_map_path, 0, 0);

    string source_label_map_path =
        TaskContext::InputFile(*task_context_.GetInput("source-label-map"));
    source_label_map_ = SharedStoreUtils::GetWithDefaultName<TermFrequencyMap>(
        source_label_map_path, 0, 0);
    string source_word_map_path =
        TaskContext::InputFile(*task_context_.GetInput("source-word-map"));
    source_word_map_ = SharedStoreUtils::GetWithDefaultName<TermFrequencyMap>(
        source_word_map_path, 0, 0);
    string source_tag_map_path =
        TaskContext::InputFile(*task_context_.GetInput("source-tag-map"));
    source_tag_map_ = SharedStoreUtils::GetWithDefaultName<TermFrequencyMap>(
        source_tag_map_path, 0, 0);
  }

  ~FeatureSize() override {
      SharedStore::Release(label_map_);
      SharedStore::Release(word_map_);
      SharedStore::Release(tag_map_);
      
      SharedStore::Release(source_label_map_);
      SharedStore::Release(source_word_map_);
      SharedStore::Release(source_tag_map_);
  }

  void Compute(OpKernelContext *context) override {
    // Computes feature sizes.
    ParserEmbeddingFeatureExtractor features(arg_prefix_);
    features.Setup(&task_context_);
    features.Init(&task_context_);
    const int num_embeddings = features.NumEmbeddings();
    Tensor *feature_sizes = nullptr;
    Tensor *domain_sizes = nullptr;
    Tensor *embedding_dims = nullptr;
    Tensor *num_actions = nullptr;
    TF_CHECK_OK(context->allocate_output(0, TensorShape({num_embeddings}),
                                         &feature_sizes));
    TF_CHECK_OK(context->allocate_output(1, TensorShape({num_embeddings}),
                                         &domain_sizes));
    TF_CHECK_OK(context->allocate_output(2, TensorShape({num_embeddings}),
                                         &embedding_dims));
    TF_CHECK_OK(context->allocate_output(3, TensorShape({}), &num_actions));
    for (int i = 0; i < num_embeddings; ++i) {
      feature_sizes->vec<int32>()(i) = features.FeatureSize(i);
      domain_sizes->vec<int32>()(i) = features.EmbeddingSize(i);
      embedding_dims->vec<int32>()(i) = features.EmbeddingDims(i);
    }

    // Computes number of actions in the transition system.
    std::unique_ptr<ParserTransitionSystem> transition_system(
        ParserTransitionSystem::Create(task_context_.Get(
            features.GetParamName("transition_system"), "arc-standard")));
    transition_system->Setup(&task_context_);
    transition_system->Init(&task_context_);

    // Note: label_map_->Size() is ignored by non-parser transition systems.
    // So even though we read the parser's label-map (output value tags and
    // their frequency), this function works for other transition systems.
    num_actions->scalar<int32>()() =
        transition_system->NumActions(word_map_->Size(), label_map_->Size());
  }

 private:
  // Task context used to configure this op.
  TaskContext task_context_;

  // Dependency label map used in transition system.
  const TermFrequencyMap *label_map_;
  const TermFrequencyMap *word_map_;
  const TermFrequencyMap *tag_map_;


  // Dependency label map used in transition system.
  const TermFrequencyMap *source_label_map_;
  const TermFrequencyMap *source_word_map_;
  const TermFrequencyMap *source_tag_map_;

  // Prefix for context parameters.
  string arg_prefix_;
};

REGISTER_KERNEL_BUILDER(Name("FeatureSize").Device(DEVICE_CPU), FeatureSize);

class FeatureVocab : public OpKernel {
 public:
  explicit FeatureVocab(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("arg_prefix", &arg_prefix_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("embedding_name", &embedding_name_));
    OP_REQUIRES_OK(context, context->MatchSignature({}, {DT_STRING}));
    LoadSpec(context, task_context_.mutable_spec());
  }

  void Compute(OpKernelContext *context) override {
    // Computes feature sizes.
    ParserEmbeddingFeatureExtractor features(arg_prefix_);
    features.Setup(&task_context_);
    features.Init(&task_context_);
    const std::vector<string> mapped_words =
        features.GetMappingsForEmbedding(embedding_name_);
    Tensor *vocab = nullptr;
    const int64 size = mapped_words.size();
    TF_CHECK_OK(context->allocate_output(0, TensorShape({size}), &vocab));
    for (int i = 0; i < size; ++i) {
      vocab->vec<string>()(i) = mapped_words[i];
    }
  }

 private:
  // Task context used to configure this op.
  TaskContext task_context_;

  // Prefix for context parameters.
  string arg_prefix_;

  // Name of embedding for which the vocabulary is to be extracted.
  string embedding_name_;
};

REGISTER_KERNEL_BUILDER(Name("FeatureVocab").Device(DEVICE_CPU), FeatureVocab);

}  // namespace
}  // namespace syntaxnet
