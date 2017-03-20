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
    : sentence_(sentence),
      transition_state_(transition_state),
      label_map_(label_map),
      word_map_(word_map),
      tag_map_(tag_map),
      stack_(-1),
      root_label_(kDefaultRootLabel) {
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
  new_state->root_label_ = root_label_;
  new_state->stack_ = stack_;
  new_state->words_.assign(words_.begin(), words_.end());
  new_state->tags_.assign(tags_.begin(), tags_.end());
  new_state->head_.assign(head_.begin(), head_.end());
  new_state->label_.assign(label_.begin(), label_.end());
  new_state->score_ = score_;
  new_state->is_gold_ = is_gold_;
  return new_state;
}

int ParserState::RootLabel() const { return root_label_; }

int ParserState::GoldWord(int position) const {
  auto word = sentence_->token(position).word();
  auto wind = word_map_->LookupIndex(word, RootLabel());
  return wind;
}

int ParserState::Word(int position) const {
  return words_[position];
}

int ParserState::StackSize() const { return stack_ + 1; }

bool ParserState::StackEmpty() const { return stack_ < 0; }

int ParserState::Head(int index) const {
  DCHECK_GE(index, -1);
  DCHECK_LT(index, head_.size());
  return index == -1 ? -1 : head_[index];
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

string ParserState::LabelAsString(int label) const {
  if (label == root_label_) return "ROOT";
  if (label >= 0 && label < label_map_->Size()) {
    return label_map_->GetTerm(label);
  }
  return "";
}

string ParserState::ToString() const {
  return transition_state_->ToString(*this);
}

}  // namespace syntaxnet
