package uk.ac.ucl.cs.mr.statnlpbook.assignment2

import scala.collection.mutable

/**
 * Created by Georgios on 05/11/2015.
 */

object Features {

  /**
   * a feature function with two templates w:word,label and l:label.
   * Example for Trigger Exraction
   * @param x
   * @param y
   * @return
   */
  def defaultTriggerFeatures(x: Candidate, y: Label): FeatureVector = {
    val doc = x.doc
    val begin = x.begin
    val end = x.end

    val thisSentence = doc.sentences(x.sentenceIndex) //use this to gain access to the parent sentence
    val feats = new mutable.HashMap[FeatureKey,Double]

    feats += FeatureKey("label bias", List(y)) -> 1.0 //bias feature
    val token = thisSentence.tokens(begin) //first token of Trigger

    feats += FeatureKey("first trigger word", List(token.word, y)) -> 1.0 //word feature
    feats.toMap
  }
  /**
   * a feature function with two templates w:word,label and l:label.
   * Example for Argument Exraction
   * @param x
   * @param y
   * @return
   */
  def defaultArgumentFeatures(x: Candidate, y: Label): FeatureVector = {
    val doc = x.doc
    val begin = x.begin
    val end = x.end
    val thisSentence = doc.sentences(x.sentenceIndex)
    val event = thisSentence.events(x.parentIndex) //use this to gain access to the parent event
    val eventHeadToken = thisSentence.tokens(event.begin) //first token of event
    val feats = new mutable.HashMap[FeatureKey,Double]
    feats += FeatureKey("label bias", List(y)) -> 1.0
    val token = thisSentence.tokens(begin) //first word of argument
    feats += FeatureKey("first argument word", List(token.word, y)) -> 1.0
    feats += FeatureKey("is protein_first trigger word", List(x.isProtein.toString,eventHeadToken.word, y)) -> 1.0
    feats.toMap
  }

  //TODO: make your own feature functions
  def myTriggerFeatures(x: Candidate, y: Label): FeatureVector = {
    val doc = x.doc
    val begin = x.begin
    val end = x.end
    val thisSentence = doc.sentences(x.sentenceIndex)
    val token = thisSentence.tokens(begin)
    val feats = new mutable.HashMap[FeatureKey,Double]

    feats += FeatureKey("label bias", List(y)) -> 1.0
    feats += FeatureKey("stem of word", List(token.stem,y)) -> 1.0
    feats += FeatureKey("pos of word", List(token.pos, y)) -> 1.0

    feats += FeatureKey("capitalisation", List(token.word.charAt(0).isUpper.toString,y)) -> 1.0


    if(begin == 0)
      feats+= FeatureKey("prior word", List(y)) -> 1.0      // if Candidate is the first word
    else {
      val prior = thisSentence.tokens(begin-1)
      feats += FeatureKey("bigram current + prior", List(token.stem,prior.word,y)) -> 1.0
    }


    if(begin == thisSentence.tokens.size-1)
      feats+= FeatureKey("next word", List(y)) -> 1.0   // if Candidate is the last word
    else {
      val next = thisSentence.tokens(begin+1)
      feats += FeatureKey("next word", List(next.word,y)) -> 1.0
    }



    val mods = thisSentence.deps.filter(e => e.mod == begin).sortBy(_.label).map(e => e.label)
    feats+= FeatureKey("mod deps", mods++List(y)) -> 1.0

    val heads = thisSentence.deps.filter(e => e.head == begin).sortBy(_.label).map(e => e.label)
    feats+= FeatureKey("head deps", heads++List(y)) -> 1.0


    feats.toMap
  }
  def myArgumentFeatures(x: Candidate, y: Label): FeatureVector = {
    ???
  }


}
