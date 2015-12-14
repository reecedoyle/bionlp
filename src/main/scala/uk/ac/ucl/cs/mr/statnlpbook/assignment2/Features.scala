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

    feats += FeatureKey("capitalisation", List(token.word.count(_.isUpper).toString,y)) -> 1.0

    if(begin == 0) {
      feats += FeatureKey("prior word", List(y)) -> 1.0 // if Candidate is the first word
    }
    else {
      val prior = thisSentence.tokens(begin-1)
      feats += FeatureKey("prior word", List(prior.stem,y)) -> 1.0
    }

    if(begin == thisSentence.tokens.size-1)
      feats+= FeatureKey("next word", List(y)) -> 1.0   // if Candidate is the last word
    else {
      val next = thisSentence.tokens(begin+1)
      feats += FeatureKey("next word", List(next.stem,y)) -> 1.0
    }

    // deps for which token is mod (going up the tree)
    val mods = thisSentence.deps.filter(e => e.mod == begin).sortBy(d => d.label+d.head+d.mod)
    for(mod <- mods){
      val mods2 = thisSentence.deps.filter(e => e.mod == mod.head).sortBy(d => d.label+d.head+d.mod)
      for(mod2 <- mods2){
        val mods3 = thisSentence.deps.filter(e => e.mod == mod2.head).sortBy(d => d.label+d.head+d.mod)
        for(mod3 <- mods3){
          feats+= FeatureKey("mod3 deps pos", List(mod3.label, thisSentence.tokens(mod3.head).pos,y)) -> 1.0
          feats+= FeatureKey("mod3 deps stem", List(mod3.label, thisSentence.tokens(mod3.head).stem,y)) -> 1.0
        }
        feats+= FeatureKey("mod2 deps pos", List(mod2.label, thisSentence.tokens(mod2.head).pos,y)) -> 1.0
        feats+= FeatureKey("mod2 deps stem", List(mod2.label, thisSentence.tokens(mod2.head).stem,y)) -> 1.0
      }
      feats+= FeatureKey("mod deps pos", List(mod.label, thisSentence.tokens(mod.head).pos,y)) -> 1.0
      feats+= FeatureKey("mod deps stem", List(mod.label, thisSentence.tokens(mod.head).stem,y)) -> 1.0
    }


    // deps for which token is head (going down the tree)
    val heads = thisSentence.deps.filter(e => e.head == begin).sortBy(d => d.label+d.head+d.mod)
    feats+= FeatureKey("head deps stem", heads.map(t => (t.label, thisSentence.tokens(t.mod).stem).toString())++List(y)) -> 1.0
    feats+= FeatureKey("head deps pos", heads.map(t => (t.label, thisSentence.tokens(t.mod).pos).toString())++List(y)) -> 1.0

    feats += FeatureKey("Proteins in sentence", List(thisSentence.mentions.size.toString,y)) -> 1.0
    feats.toMap
  }

  def myTriggerFeaturesNB(x: Candidate, y: Label) : FeatureVector = {
    val doc = x.doc
    val begin = x.begin
    val end = x.end
    val thisSentence = doc.sentences(x.sentenceIndex)
    val token = thisSentence.tokens(begin)
    val feats = new mutable.HashMap[FeatureKey,Double]

    feats += FeatureKey("label bias", List(y)) -> 1.0
    feats += FeatureKey("stem of word", List(token.stem,y)) -> 1.0
//    feats += FeatureKey("pos of word", List(token.pos, y)) -> 1.0

    feats += FeatureKey("capitalisation", List(token.word.count(_.isUpper).toString,y)) -> 1.0

    feats.toMap
  }
  def myArgumentFeatures(x: Candidate, y: Label): FeatureVector = {
    val doc = x.doc
    val begin = x.begin
    val end = x.end
    val thisSentence = doc.sentences(x.sentenceIndex)
    val token = thisSentence.tokens(begin)
    val parentToken = thisSentence.tokens(x.parentIndex)
    val feats = new mutable.HashMap[FeatureKey,Double]

    /*
    feats += FeatureKey("label bias", List(y)) -> 1.0


    feats += FeatureKey("Proteins in sentence", List(thisSentence.mentions.size.toString, y)) -> 1.0    //Number of proteins in the sentence

//    feats += FeatureKey("capitalisation", List(token.word.count(_.isUpper).toString,y)) -> 1.0  // number of captialisations in the argument

    // Bigram (each way)
    if(begin == 0) {
      feats += FeatureKey("prior word", List(y)) -> 1.0 // if Candidate is the first word
    }
    else {
      val prior = thisSentence.tokens(begin-1)
      feats += FeatureKey("prior word", List(prior.stem,y)) -> 1.0
    }

    if(begin == thisSentence.tokens.size-1)
      feats+= FeatureKey("next word", List(y)) -> 1.0   // if Candidate is the last word
    else {
      val next = thisSentence.tokens(begin+1)
      feats += FeatureKey("next word", List(next.stem,y)) -> 1.60
    }

    // Trigram prior (the 'begin == 0' case is handled above)
    if (begin >= 2) {
      feats += FeatureKey("prior words 3", List(thisSentence.tokens(begin - 2).stem, thisSentence.tokens(begin - 1).stem, y)) -> 0
    }

    // Trigram next
    if (begin < thisSentence.tokens.size - 3) {
      feats += FeatureKey("next words 3", List(thisSentence.tokens(begin + 1).stem, thisSentence.tokens(begin + 2).stem, y)) -> 0
    }

    feats += FeatureKey("is protein", List(x.isProtein.toString, y)) -> 1.0

    // Getting edges
    val candidateEdges = thisSentence.events.filter(e => e.arguments.contains(token.word))
    val candidateIndicies = candidateEdges.map(e => e.sentenceIndex)
    val candidateWords = candidateIndicies.map(e => thisSentence.tokens(e).stem).sorted //tokens(e).stem
    val candidatePos = candidateIndicies.map(e => thisSentence.tokens(e).pos)

    feats += FeatureKey("tokens pointing to argument", candidatePos ++ List(y)) -> 1.0

//    feats += FeatureKey("number of tokens pointing to argument", List(candidateIndicies.size.toString, y)) -> 1.0
*/


    feats += FeatureKey("label bias", List(y)) -> 1.0

  //                                 Lexical
  // -----------------------------------------------------------------------------
    //feats += FeatureKey("word of candidate and parentEvent", List(token.word, parentToken.word,y)) -> 1.0
    //feats += FeatureKey("word of candidate", List(token.word,y)) -> 1.0
    //feats += FeatureKey("word of parentEvent", List(parentToken.word,y)) -> 1.0
     // feats += FeatureKey("capitalisation of candidate", List(token.word(0).isUpper.toString, y)) -> 1.0
      feats += FeatureKey("pos of parentEvent", List(parentToken.pos,y)) -> 1.0
      feats += FeatureKey("number of capitalised letters", List(token.word.count(_.isUpper).toString,y)) -> 1.0
   //   feats+= FeatureKey("candidate has hifen", List(token.word.contains("-").toString,y)) -> 1.0
  // -----------------------------------------------------------------------------



    //                                Entity
    // -----------------------------------------------------------------------------
   // feats += FeatureKey("Proteins in sentence", List(thisSentence.mentions.size.toString, y)) -> 1.0
      feats += FeatureKey("candidate is protein", List(x.isProtein.toString,y)) -> 1.0
    //  feats+= FeatureKey("Protein Hifen", List(token.word.contains("-").toString,x.isProtein.toString,y)) -> 1.0
    // -----------------------------------------------------------------------------



    //                                 Syntax
    // -----------------------------------------------------------------------------
    //val heads = thisSentence.deps.filter(e => e.head == begin).sortBy(d => d.label+d.head+d.mod)
    //feats+= FeatureKey("head deps stem", heads.map(t => (thisSentence.tokens(t.mod).stem).toString())++List(y)) -> 1.0
    // -----------------------------------------------------------------------------



    //                                 Other
    // -----------------------------------------------------------------------------
    //feats += FeatureKey("pos and Isprotein", List(token.pos,x.isProtein.toString,y)) -> 1.0
    //feats+= FeatureKey("absolute distance from candidate", List(Math.abs(x.begin - x.parentIndex).toString,y)) -> 1.0
   // feats += FeatureKey("Parentcapitalisation count", List(parentToken.word.count(_.isUpper).toString,y)) -> 1.0
   // feats += FeatureKey("capitalisation count", List(token.word.count(_.isUpper).toString,y)) -> 1.0
    //feats+= FeatureKey("isProtein and word of parentIndex", List(x.isProtein.toString, parentToken.stem)) -> 1.0
   // feats+= FeatureKey("contains regulation keyword", List(parentToken.word.toLowerCase().contains("reg").toString, y)) -> 1.0
    // -----------------------------------------------------------------------------
















    feats.toMap
  }


}
