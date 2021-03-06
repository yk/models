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

// Arc-standard transition system.
//
// This transition system has three types of actions:
//  - The SHIFT action pushes the next input token to the stack and
//    advances to the next input token.
//  - The LEFT_ARC action adds a dependency relation from first to second token
//    on the stack and removes second one.
//  - The RIGHT_ARC action adds a dependency relation from second to first token
//    on the stack and removes the first one.
//
// The transition system operates with parser actions encoded as integers:
//  - A SHIFT action is encoded as 0.
//  - A LEFT_ARC action is encoded as an odd number starting from 1.
//  - A RIGHT_ARC action is encoded as an even number starting from 2.

#include <string>

#include "syntaxnet/parser_state.h"
#include "syntaxnet/parser_transitions.h"
#include "syntaxnet/utils.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include <iostream>

namespace syntaxnet {

class ArcStandardTransitionState : public ParserTransitionState {
 public:
     const ParserTransitionSystem* transition_system_;
  // Clones the transition state by returning a new object.
  ParserTransitionState *Clone() const override {
    auto* tst = new ArcStandardTransitionState();
    tst->transition_system_ = transition_system_;
    return tst;
  }

  // Pushes the root on the stack before using the parser state in parsing.
  void Init(ParserState *state) override { 
      state->stack_ = 0;
      state->head_.push_back(-1);
      state->label_.push_back(state->RootLabel());
  }

  // Adds transition state specific annotations to the document.
  void AddParseToDocument(const ParserState &state, bool rewrite_root_labels,
                          Sentence *sentence) const override {
    sentence->clear_token();
    for (int i = 0; i < state.NumTokens(); ++i) {
      Token *token = sentence->add_token();
      token->set_word(state.WordAsString(state.Word(i)));
      token->set_label(state.LabelAsString(state.Label(i)));
      if (state.Head(i) != -1) {
        token->set_head(state.Head(i));
      } else {
        token->clear_head();
        if (rewrite_root_labels) {
          token->set_label(state.LabelAsString(state.RootLabel()));
        }
      }
    }
    sentence->clear_transition();
    for(auto& tr : state.transitions_){
        auto* ntr = sentence->add_transition();
        ntr->set_action(tr);
        ntr->set_label(transition_system_->ActionAsString(tr, state));
    }
  }
  // Whether a parsed token should be considered correct for evaluation.
  bool IsTokenCorrect(const ParserState &state, int index) const override {
    auto a = state.GoldHead(index) == state.Head(index);

    auto windex = state.stack_ + state.words_.size() - index - 1;
    auto b = state.Word(windex) == state.GoldWord(windex);

    return a && b;
  }

  // Returns a human readable string representation of this state.
  string ToString(const ParserState &state) const override {
    string str;
    for (int i = 0; i < state.NumTokens(); ++i) {
      tensorflow::strings::StrAppend(&str, " ", std::to_string(state.Word(i)));
    }
    return str;
  }
};

class ArcStandardTransitionSystem : public ParserTransitionSystem {
 public:
  // Action types for the arc-standard transition system.
  enum ParserActionType {
    SHIFT = 0,
    LEFT_ARC = 1,
    RIGHT_ARC = 2,
  };

  // The SHIFT action uses the same value as the corresponding action type.
  static ParserAction ShiftAction(int word, const ParserState* state) { 
      return word + 2 * state->NumLabels();
  }

  // The LEFT_ARC action converts the label to an odd number greater or equal
  // to 1.
  static ParserAction LeftArcAction(int label, const ParserState* state) { 
      return label;
  }

  // The RIGHT_ARC action converts the label to an even number greater or equal
  // to 2.
  static ParserAction RightArcAction(int label, const ParserState* state) {
    //return 1 + ((label << 1) | 1);
    return label + state->NumLabels();
  }

  // Extracts the action type from a given parser action.
  static ParserActionType ActionType(ParserAction action, const ParserState* state) {
      auto nl = state->NumLabels();
      if(action < nl){
          return LEFT_ARC;
      }else if(action < 2*nl){
          return RIGHT_ARC;
      }else{
          return SHIFT;
      }
  }

  // Extracts the label from a given parser action. If the action is SHIFT,
  // returns -1.
  static int Label(ParserAction action, const ParserState* state) {
      auto nl = state->NumLabels();
      if(action < nl)
          return action;
      action -= nl;
      if(action < nl)
          return action;
      action -= nl;
      return action;
  }

  // Returns the number of action types.
  int NumActionTypes() const override { return 3; }

  // Returns the number of possible actions.
  int NumActions(int num_words, int num_labels) const override {
      return 1 * num_words + 2 * num_labels;
  }

  // The method returns the default action for a given state.
  ParserAction GetDefaultAction(const ParserState &state) const override {
    // If there are further tokens available in the input then Shift.
    //if (!state.EndOfInput()) return ShiftAction();

    // Do a "reduce".
    return ShiftAction(0, &state);
  }

  // Returns the next gold action for a given state according to the
  // underlying annotated sentence.
  ParserAction GetNextGoldAction(const ParserState &state) const override {
    //longest right arc if possible
    //shortest left arc if possible
    //shift else
      auto goldIndex = state.GoldIndex();
      //std::cerr << "GI: " << goldIndex << ", stack: " << state.stack_ << std::endl;
      auto rightArc = state.GoldRightArcBeforeBuffer(goldIndex);
      if(rightArc != -1 && IsAllowedRightArc(state)){
          //std::cerr << "GoldRight " << goldIndex << " -> " << rightArc << std::endl;
          return RightArcAction(rightArc, &state);
      }
      
      auto leftArc = state.GoldLeftArcBeforeBuffer(goldIndex);
      if(leftArc != -1 && IsAllowedLeftArc(state)){
          //std::cerr << "GoldLeft " << goldIndex << " -> " << leftArc << std::endl;
          return LeftArcAction(leftArc, &state);
      }

      auto goldWord = state.GoldWord(goldIndex);
      //std::cerr << "GoldShift " << goldIndex << " : " << goldWord << std::endl;
      return ShiftAction(goldWord, &state);


    //// If the stack contains less than 2 tokens, the only valid parser action is
    //// shift.
    //if (state.StackSize() < 2) {
      ////DCHECK(!state.EndOfInput());
      //return ShiftAction(0);
    //}

    //// If the second token on the stack is the head of the first one,
    //// return a right arc action.
    //if (state.GoldHead(state.Stack(0)) == state.Stack(1) &&
        //DoneChildrenRightOf(state, state.Stack(0))) {
      //const int gold_label = state.GoldLabel(state.Stack(0));
      //return RightArcAction(gold_label);
    //}

    //// If the first token on the stack is the head of the second one,
    //// return a left arc action.
    //if (state.GoldHead(state.Stack(1)) == state.Top()) {
      //const int gold_label = state.GoldLabel(state.Stack(1));
      //return LeftArcAction(gold_label);
    //}

    // Otherwise, shift.
  }

  //// Determines if a token has any children to the right in the sentence.
  //// Arc standard is a bottom-up parsing method and has to finish all sub-trees
  //// first.
  //static bool DoneChildrenRightOf(const ParserState &state, int head) {
    //int index = state.Next();
    //int num_tokens = state.sentence().token_size();
    //while (index < num_tokens) {
      //// Check if the token at index is the child of head.
      //int actual_head = state.GoldHead(index);
      //if (actual_head == head) return false;

      //// If the head of the token at index is to the right of it there cannot be
      //// any children in-between, so we can skip forward to the head.  Note this
      //// is only true for projective trees.
      //if (actual_head > index) {
        //index = actual_head;
      //} else {
        //++index;
      //}
    //}
    //return true;
  //}

  // Checks if the action is allowed in a given parser state.
  bool IsAllowedAction(ParserAction action,
                       const ParserState &state) const override {
    switch (ActionType(action, &state)) {
      case SHIFT:
        return IsAllowedShift(state);
      case LEFT_ARC:
        return IsAllowedLeftArc(state);
      case RIGHT_ARC:
        return IsAllowedRightArc(state);
    }

    return false;
  }

  // Returns true if a shift is allowed in the given parser state.
  bool IsAllowedShift(const ParserState &state) const {
    // We can shift if there are more input tokens.
    return !state.StackEmpty();
  }

  // Returns true if a left-arc is allowed in the given parser state.
  bool IsAllowedLeftArc(const ParserState &state) const {
    // Left-arc requires two or more tokens on the stack but the first token
    // is the root an we do not want and left arc to the root.
    return !state.StackEmpty() && state.head_.size() < 20;
  }

  // Returns true if a right-arc is allowed in the given parser state.
  bool IsAllowedRightArc(const ParserState &state) const {
    // Right arc requires three or more tokens on the stack.
    return !state.StackEmpty() && state.head_.size() < 20;
  }

  // Performs the specified action on a given parser state, without adding the
  // action to the state's history.
  void PerformActionWithoutHistory(ParserAction action,
                                   ParserState *state) const override {
    switch (ActionType(action, state)) {
      case SHIFT:
        PerformShift(state, Label(action, state));
        break;
      case LEFT_ARC:
        PerformLeftArc(state, Label(action, state));
        break;
      case RIGHT_ARC:
        PerformRightArc(state, Label(action, state));
        break;
    }
  }

  // Makes a shift by pushing the next input token on the stack and moving to
  // the next position.
  void PerformShift(ParserState *state, int word) const {
    //std::cerr << "Shift " << word << std::endl;
    DCHECK(IsAllowedShift(*state));
    state->words_.push_back(word);
    state->stack_--;
    state->transitions_.push_back(ShiftAction(word, state));
  }

  // Makes a left-arc between the two top tokens on stack and pops the second
  // token on stack.
  void PerformLeftArc(ParserState *state, int label) const {
    //std::cerr << "LeftArc " << label << std::endl;
    DCHECK(IsAllowedLeftArc(*state));
    //int head = state->head_.at(state->stack_);
    int head = state->stack_;
    state->label_.insert(state->label_.begin() + state->stack_, label);
    state->head_.insert(state->head_.begin() + state->stack_, head);
    state->stack_++;
    state->transitions_.push_back(LeftArcAction(label, state));
  }

  // Makes a right-arc between the two top tokens on stack and pops the stack.
  void PerformRightArc(ParserState *state, int label) const {
    //std::cerr << "RightArc " << label << std::endl;
    DCHECK(IsAllowedRightArc(*state));
    //int head = state->head_.at(state->stack_);
    int head = state->stack_;
    state->stack_++;
    state->label_.insert(state->label_.begin() + state->stack_, label);
    state->head_.insert(state->head_.begin() + state->stack_, head);
    state->transitions_.push_back(RightArcAction(label, state));
  }

  // We are in a deterministic state when we either reached the end of the input
  // or reduced everything from the stack.
  bool IsDeterministicState(const ParserState &state) const override {
    return state.StackSize() < 1;
    //return state.StackSize() < 2 && !state.EndOfInput();
  }

  // We are in a final state when we reached the end of the input and the stack
  // is empty.
  bool IsFinalState(const ParserState &state) const override {
    //return state.EndOfInput() && state.StackSize() < 2;
    return state.StackSize() < 1;
  }

  // Returns a string representation of a parser action.
  string ActionAsString(ParserAction action,
                        const ParserState &state) const override {
    switch (ActionType(action, &state)) {
      case SHIFT:
        return "SHIFT(" + state.WordAsString(Label(action, &state)) + ":" + std::to_string(Label(action, &state)) + ")";
      case LEFT_ARC:
        return "LEFT_ARC(" + state.LabelAsString(Label(action, &state)) + ":" + std::to_string(Label(action, &state)) + ")";
      case RIGHT_ARC:
        return "RIGHT_ARC(" + state.LabelAsString(Label(action, &state)) + ":" + std::to_string(Label(action, &state)) + ")";
    }
    return "UNKNOWN";
  }

  // Returns a new transition state to be used to enhance the parser state.
  ParserTransitionState *NewTransitionState(bool training_mode) const override {
    auto* tst = new ArcStandardTransitionState();
    tst->transition_system_ = this;
    return tst;
  }

  // Meta information API. Returns token indices to link parser actions back
  // to positions in the input sentence.
  bool SupportsActionMetaData() const override { return true; }

  //// Returns the child of a new arc for reduce actions.
  //int ChildIndex(const ParserState &state,
                 //const ParserAction &action) const override {
    //switch (ActionType(action)) {
      //case SHIFT:
        //return -1;
      //case LEFT_ARC:  // left arc pops stack(1)
        //return state.Stack(1);
      //case RIGHT_ARC:
        //return state.Stack(0);
      //default:
        //LOG(FATAL) << "Invalid parser action: " << action;
    //}
  //}

  //// Returns the parent of a new arc for reduce actions.
  //int ParentIndex(const ParserState &state,
                  //const ParserAction &action) const override {
    //switch (ActionType(action)) {
      //case SHIFT:
        //return -1;
      //case LEFT_ARC:  // left arc pops stack(1)
        //return state.Stack(0);
      //case RIGHT_ARC:
        //return state.Stack(1);
      //default:
        //LOG(FATAL) << "Invalid parser action: " << action;
    //}
  //}

};

REGISTER_TRANSITION_SYSTEM("arc-standard", ArcStandardTransitionSystem);

}  // namespace syntaxnet
