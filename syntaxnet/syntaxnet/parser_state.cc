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

#include "syntaxnet/parser_state.h"

#include <iostream>
#include "syntaxnet/kbest_syntax.pb.h"
#include "syntaxnet/sentence.pb.h"
#include "syntaxnet/term_frequency_map.h"
#include "syntaxnet/utils.h"

namespace syntaxnet {

const char ParserState::kRootLabel[] = "ROOT";

ParserState::ParserState(Sentence *sentence,
                         ParserTransitionState *transition_state,
                         const TermFrequencyMap *label_map,
                         const TermFrequencyMap *word_map,
                         const TermFrequencyMap *tag_map
                         )
    : stack_(-1),
      sentence_(sentence),
      transition_state_(transition_state),
      label_map_(label_map),
      word_map_(word_map),
      tag_map_(tag_map),
      root_label_(label_map->LookupIndex(kRootLabel,
                                         kDefaultRootLabel  /* unknown */)){
  // Initialize the stack. Some transition systems could also push the
  // artificial root on the stack, so we make room for that as well.

  // Allocate space for head indices and labels. Initialize the head for all
  // tokens to be the artificial root node, i.e. token -1.

  // Transition system-specific preprocessing.
  if (transition_state_ != nullptr) transition_state_->Init(this);
}

ParserState::~ParserState() { delete transition_state_; }

ParserState *ParserState::Clone() const {
  ParserState *new_state = new ParserState();
  new_state->sentence_ = sentence_;
  new_state->alternative_ = alternative_;
  new_state->transition_state_ =
      (transition_state_ == nullptr ? nullptr : transition_state_->Clone());
  new_state->label_map_ = label_map_;
  new_state->word_map_ = word_map_;
  new_state->tag_map_ = tag_map_;
  new_state->root_label_ = root_label_;
  new_state->stack_ = stack_;
  new_state->words_.assign(words_.begin(), words_.end());
  new_state->tags_.assign(tags_.begin(), tags_.end());
  new_state->head_.assign(head_.begin(), head_.end());
  new_state->label_.assign(label_.begin(), label_.end());
  new_state->transitions_.assign(transitions_.begin(), transitions_.end());
  new_state->score_ = score_;
  new_state->is_gold_ = is_gold_;
  new_state->keep_history_ = keep_history_;
  new_state->history_.assign(history_.begin(), history_.end());
  return new_state;
}

int ParserState::RootLabel() const { return root_label_; }

int ParserState::NthArc(int head, int arc) const {
    auto& t = sentence_->target();
    int inc = arc > 0 ? 1 : -1;
    arc -= inc;
    for(int h = head + inc; h < t.token_size() && h >= 0; h += inc){
        if(t.token(h).head() == head){
            if(arc == 0){
                return h;
            }
            arc -= inc;
        }
    }
    return -1;
}

int ParserState::NumWords() const { return word_map_->Size(); }
int ParserState::NumTags() const { return tag_map_->Size(); }

int ParserState::NthArcN(int head, int node) const {
    DCHECK(head_[node] == head);
    int n = 0;
    int inc = node > head ? 1 : -1;
    for(int h = head + inc; h * inc < node * inc; h += inc){
        if(head_[h] == head){
            n += inc;
        }
    }
    return n + inc;
}

int ParserState::GoldIndex() const {
    DCHECK_GE(stack_, 0);
    auto s = stack_;
    //std::cerr << "labels";
    //for(auto& n : label_){
        //std::cerr << " " << n;
    //}
    //std::cerr << std::endl;
    vector<int> ns;
    while(s >= 0 && s < label_.size() && head_[s] != s && label_[s] != RootLabel()){
        int n = NthArcN(head_[s], s);
        ns.push_back(n);
        s = head_[s];
    }
    auto g = GoldRoot();
    //std::cerr << "GOLDROOT: " << g << std::endl;
    while(g != -1 && ns.size() > 0){
        int nb = ns.back();
        auto new_g = NthArc(g, nb);
        if(new_g == -1)
            break;
        g = new_g;
        ns.pop_back();
    }
    if(g == -1){
        g = sentence_->target().token_size() - (head_.size() - 1 - stack_) - 1;
        //std::cerr <<sentence_->target().token_size() << " " << head_.size() << " " << stack_ << " "<< g << std::endl;
        //std::cerr << "case 1" << std::endl;
    }
    if(g < 0){
        g = 0;
        //std::cerr << "case 2" << std::endl;
    }
    if(g >= sentence_->target().token_size()){
        g = sentence_->target().token_size() - 1;
        //std::cerr << "case 3" << std::endl;
    }
    //std::cerr << "G: " << g << std::endl;
    return g;
}

int ParserState::GoldLeftArcBeforeBuffer(int goldIndex) const {
    auto& t = sentence_->target();
    auto bufSize = head_.size() - 1 - stack_;
    auto goldStackSize = t.token_size() - bufSize;
    if(goldIndex < goldStackSize){
        goldStackSize = goldIndex;
    }
    for(int g = goldStackSize - 1; g >= 0; g--){
        auto& tt = t.token(g);
        if(tt.head() == goldIndex){
            auto l = tt.label();
            return label_map_->LookupIndex(l, RootLabel());
        }
    }
    return -1;
}

int ParserState::GoldRightArcBeforeBuffer(int goldIndex) const {
    auto& t = sentence_->target();
    auto bufSize = head_.size() - 1 - stack_;
    auto goldStackSize = t.token_size() - bufSize;
    for(int g = goldStackSize - 1; g > goldIndex; g--){
        auto& tt = t.token(g);
        if(tt.head() == goldIndex){
            auto l = tt.label();
            return label_map_->LookupIndex(l, RootLabel());
        }
    }
    return -1;
}

int ParserState::GoldRoot() const {
    auto& t = sentence_->target();
    for(int i=0; i<t.token_size(); i++){
        if(t.token(i).head() == -1){
            return i;
        }
    }
    return -1;
}

int ParserState::GoldWord(int position) const {
  auto word = sentence_->target().token(position).word();
  auto wind = word_map_->LookupIndex(word, word_map_->Size() - 1);
  return wind;
}

int ParserState::Word(int position) const {
  return words_[NumTokens() - 1 - position];
int ParserState::NumLabels() const {
  return label_map_->Size() + (RootLabel() == kDefaultRootLabel ? 1 : 0);
}

int ParserState::StackSize() const { return stack_ + 1; }

bool ParserState::StackEmpty() const { return stack_ < 0; }

int ParserState::Stack(int position) const {
  return stack_ - position;
}


int ParserState::Head(int index) const {
  DCHECK_GE(index, -1);
  DCHECK_LT(index, head_.size());
  return index == -1 ? -1 : head_[index];
}

int ParserState::SourceHead(int index) const {
  DCHECK_GE(index, -1);
  auto& src = sentence().source();
  DCHECK_LT(index, src.token_size());
  return index == -1 ? -1 : src.token(index).head();
}

int ParserState::Label(int index) const {
  DCHECK_GE(index, -1);
  DCHECK_LT(index, head_.size());
  return index == -1 ? RootLabel() : label_[index];
}

int ParserState::Parent(int index, int n) const {
  // Find the n-th parent by applying the head function n times.
  DCHECK_GE(index, -1);
  DCHECK_LT(index, head_.size());
  while (n-- > 0) index = Head(index);
  return index;
}

int ParserState::LeftmostChild(int index, int n) const {
  DCHECK_GE(index, -1);
  DCHECK_LT(index, head_.size());
  while (n-- > 0) {
    // Find the leftmost child by scanning from start until a child is
    // encountered.
    int i;
    for (i = -1; i < index; ++i) {
      if (Head(i) == index) break;
    }
    if (i == index) return -2;
    index = i;
  }
  return index;
}

int ParserState::RightmostChild(int index, int n) const {
  DCHECK_GE(index, -1);
  DCHECK_LT(index, head_.size());
  while (n-- > 0) {
    // Find the rightmost child by scanning backward from end until a child
    // is encountered.
    int i;
    for (i = head_.size() - 1; i > index; --i) {
      if (Head(i) == index) break;
    }
    if (i == index) return -2;
    index = i;
  }
  return index;
}

int ParserState::LeftSibling(int index, int n) const {
  // Find the n-th left sibling by scanning left until the n-th child of the
  // parent is encountered.
  DCHECK_GE(index, -1);
  DCHECK_LT(index, head_.size());
  if (index == -1 && n > 0) return -2;
  int i = index;
  while (n > 0) {
    --i;
    if (i == -1) return -2;
    if (Head(i) == Head(index)) --n;
  }
  return i;
}

int ParserState::RightSibling(int index, int n) const {
  // Find the n-th right sibling by scanning right until the n-th child of the
  // parent is encountered.
  DCHECK_GE(index, -1);
  DCHECK_LT(index, head_.size());
  if (index == -1 && n > 0) return -2;
  int i = index;
  while (n > 0) {
    ++i;
    if (i == head_.size()) return -2;
    if (Head(i) == Head(index)) --n;
  }
  return i;
}

void ParserState::AddArc(int index, int head, int label) {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, head_.size());
  head_[index] = head;
  label_[index] = label;
}

int ParserState::GoldHead(int index) const {
  // A valid ParserState index is transformed to a valid Sentence index,
  // then the gold head is extracted.
  DCHECK_GE(index, -1);
  DCHECK_LT(index, head_.size());
  if (index == -1) return -1;
  const int offset = 0;
  const int gold_head = GetToken(index).head();
  return gold_head == -1 ? -1 : gold_head - offset;
}

int ParserState::GoldLabel(int index) const {
  // A valid ParserState index is transformed to a valid Sentence index,
  // then the gold label is extracted.
  DCHECK_GE(index, -1);
  DCHECK_LT(index, head_.size());
  if (index == -1) return RootLabel();
  string gold_label;
  gold_label = GetToken(index).label();
  return label_map_->LookupIndex(gold_label, RootLabel() /* unknown */);
}

void ParserState::AddParseToDocument(Sentence *sentence,
                                     bool rewrite_root_labels) const {
  transition_state_->AddParseToDocument(*this, rewrite_root_labels, sentence);
}

bool ParserState::IsTokenCorrect(int index) const {
  return transition_state_->IsTokenCorrect(*this, index);
}

string ParserState::WordAsString(int word) const {
  if (word < 0) return "UNK";
  if (word >= 0 && word < word_map_->Size()) {
    return word_map_->GetTerm(word);
  }
  return "";
}

string ParserState::LabelAsString(int label) const {
  if (label == root_label_) return kRootLabel;
  if (label >= 0 && label < label_map_->Size()) {
    return label_map_->GetTerm(label);
  }
  return "";
}

string ParserState::ToString() const {
  return transition_state_->ToString(*this);
}

}  // namespace syntaxnet
