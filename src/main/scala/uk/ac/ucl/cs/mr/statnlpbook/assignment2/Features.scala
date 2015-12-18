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
    val feats = new mutable.HashMap[FeatureKey, Double]

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
    val feats = new mutable.HashMap[FeatureKey, Double]
    feats += FeatureKey("label bias", List(y)) -> 1.0
    val token = thisSentence.tokens(begin) //first word of argument
    feats += FeatureKey("first argument word", List(token.word, y)) -> 1.0
    feats += FeatureKey("is protein_first trigger word", List(x.isProtein.toString, eventHeadToken.word, y)) -> 1.0
    feats.toMap
  }

  //TODO: make your own feature functions
  def myTriggerFeatures(x: Candidate, y: Label): FeatureVector = {
    val doc = x.doc
    val begin = x.begin
    val end = x.end
    val thisSentence = doc.sentences(x.sentenceIndex)
    val parent = x.parentIndex
    val token = thisSentence.tokens(begin)
    val feats = new mutable.HashMap[FeatureKey, Double]

    feats += FeatureKey("label bias", List(y)) -> 1.0
    feats += FeatureKey("stem of word", List(token.stem, y)) -> 1.0
    feats += FeatureKey("pos of word", List(token.pos, y)) -> 1.0

    feats += FeatureKey("capitalisation", List(token.word.count(_.isUpper).toString, y)) -> 1.0

    if (begin == 0) {
      feats += FeatureKey("prior word", List(y)) -> 1.0 // if Candidate is the first word
    }
    else {
      val prior = thisSentence.tokens(begin - 1)
      feats += FeatureKey("prior word", List(prior.stem, y)) -> 1.0
    }

    if (begin == thisSentence.tokens.size - 1)
      feats += FeatureKey("next word", List(y)) -> 1.0 // if Candidate is the last word
    else {
      val next = thisSentence.tokens(begin + 1)
      feats += FeatureKey("next word", List(next.stem, y)) -> 1.0
    }

    // deps for which token is mod (going up the tree)
    val mods = thisSentence.deps.filter(e => e.mod == begin).sortBy(d => d.label + d.head + d.mod)
    for (mod <- mods) {
      val mods2 = thisSentence.deps.filter(e => e.mod == mod.head).sortBy(d => d.label + d.head + d.mod)
      for (mod2 <- mods2) {
        val mods3 = thisSentence.deps.filter(e => e.mod == mod2.head).sortBy(d => d.label + d.head + d.mod)
        for (mod3 <- mods3) {
          feats += FeatureKey("mod3 deps pos", List(mod3.label, thisSentence.tokens(mod3.head).pos, y)) -> 1.0
          feats += FeatureKey("mod3 deps stem", List(mod3.label, thisSentence.tokens(mod3.head).stem, y)) -> 1.0
        }
        feats += FeatureKey("mod2 deps pos", List(mod2.label, thisSentence.tokens(mod2.head).pos, y)) -> 1.0
        feats += FeatureKey("mod2 deps stem", List(mod2.label, thisSentence.tokens(mod2.head).stem, y)) -> 1.0
      }
      feats += FeatureKey("mod deps pos", List(mod.label, thisSentence.tokens(mod.head).pos, y)) -> 1.0
      feats += FeatureKey("mod deps stem", List(mod.label, thisSentence.tokens(mod.head).stem, y)) -> 1.0
    }


    // deps for which token is head (going down the tree)
    val heads = thisSentence.deps.filter(e => e.head == begin).sortBy(d => d.label + d.head + d.mod)
    feats += FeatureKey("head deps stem", heads.map(t => (t.label, thisSentence.tokens(t.mod).stem).toString()) ++ List(y)) -> 1.0
    feats += FeatureKey("head deps pos", heads.map(t => (t.label, thisSentence.tokens(t.mod).pos).toString()) ++ List(y)) -> 1.0

    feats += FeatureKey("Proteins in sentence", List(thisSentence.mentions.size.toString, y)) -> 1.0
    feats.toMap
  }

  def myTriggerFeaturesNB(x: Candidate, y: Label): FeatureVector = {
    val doc = x.doc
    val begin = x.begin
    val end = x.end
    val thisSentence = doc.sentences(x.sentenceIndex)
    val token = thisSentence.tokens(begin)
    val feats = new mutable.HashMap[FeatureKey, Double]

    feats += FeatureKey("label bias", List(y)) -> 1.0
    feats += FeatureKey("stem of word", List(token.stem, y)) -> 1.0
    //  feats += FeatureKey("pos of word", List(token.pos, y)) -> 1.0

    feats += FeatureKey("capitalisation", List(token.word.count(_.isUpper).toString, y)) -> 1.0

    feats.toMap
  }

  def myArgumentFeatures(x: Candidate, y: Label): FeatureVector = {
    val doc = x.doc
    val begin = x.begin
    val end = x.end
    val thisSentence = doc.sentences(x.sentenceIndex)
    val token = thisSentence.tokens(begin)
    val parentCandidate = thisSentence.events(x.parentIndex)
    val parentToken = thisSentence.tokens(parentCandidate.begin)
    val heads = thisSentence.deps.filter(e => e.head == begin)
    val mods = thisSentence.deps.filter(e => e.mod == begin)
    //val parentArgsCount = parentCandidate.arguments.filter(e=> e.gold != "None").size

    //println(parentCandidate)
    // println(parentCandidate.arguments.filter(e=> e.gold != "None"))

    /*if (y == "Theme" || y == "Cause")
    println ("Type: " + y + ", token.index = " + token.index + ", x.parentIndex = " + x.parentIndex + ", distance = " + Math.abs(token.index - x.parentIndex))*/


    val feats = new mutable.HashMap[FeatureKey, Double]

    /*
    feats += FeatureKey("label bias", List(y)) -> 1.0


    feats += FeatureKey("Proteins in sentence", List(thisSentence.mentions.size.toString, y)) -> 1.0    //Number of proteins in the sentence

//    feats += FeatureKey("capitalisation", List(token.word.count(_.isUpper).toString,y)) -> 1.0  // number of captialisations in the argument

    // Lexical context (each way)
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

    feats += FeatureKey("Arg Label bias", List(y)) -> 1.0

    //                                 Lexical
    // -----------------------------------------------------------------------------
    feats += FeatureKey("Arg pos of parent and candidate are equal", List((token.pos == parentToken.pos).toString, y)) -> 1.0 // helps both. generally helps argument extraction
    feats += FeatureKey("Arg POS and parent POS", List(token.pos, parentToken.pos, y)) -> 1.0

    //feats += FeatureKey("Arg POS", List(token.pos, y)) -> 1.0 // VERY good at telling when argument is none and when it is NOT.
    //feats += FeatureKey("Arg POS and parent POS", List(token.pos, parentToken.pos, y)) -> 1.0
    //feats += FeatureKey("Arg Word", List(token.word, y)) -> 1.0
    //feats += FeatureKey("Arg Stem", List(token.stem, y)) -> 1.0
    //feats += FeatureKey("Arg value of word", List(token.word, y)) -> 1.0 // really good at telling when something is Cause and very good at telling when something is NOT none
    //feats += FeatureKey("Arg POS = NN, isProtein", List((token.pos == "NN" && x.isProtein).toString, y)) -> 1.0

    //Stems
    feats += FeatureKey("Arg Stem of candidate and parent", List(parentToken.stem, token.stem, y)) -> 1.0
    //feats += FeatureKey("Arg Stem of Parent", List(parentToken.stem, y)) -> 1.0
    //feats += FeatureKey("Arg Stem", List(token.stem, y)) -> 1.0

    feats += FeatureKey("Arg Stem of Parent and isProtein", List(parentToken.stem, x.isProtein.toString, y)) -> 1.0
    //feats += FeatureKey("word contains reg", List(token.word.toLowerCase.contains("reg").toString,y)) -> 1.0
    //feats += FeatureKey("word of candidate and parentEvent", List(token.word, parentToken.word,y)) -> 1.0
    //feats += FeatureKey("word of candidate", List(token.word,y)) -> 1.0
    //feats += FeatureKey("word of parentEvent", List(parentToken.word,y)) -> 1.0
    feats += FeatureKey("Arg capitalisation of candidate and is protein", List(token.word.exists(_.isUpper).toString, x.isProtein.toString, y)) -> 1.0  // ability to classify theme goes down but cause goes up.
    //feats += FeatureKey("pos of parentEvent", List(parentToken.pos,y)) -> 1.0
    //feats += FeatureKey("number of capitalised letters", List(token.word.count(_.isUpper).toString,y)) -> 1.0
    //feats += FeatureKey("candidate has hifen", List(token.word.contains("-").toString,y)) -> 1.0
    // -----------------------------------------------------------------------------

    //                                Entity
    // -----------------------------------------------------------------------------
    feats += FeatureKey("Proteins in sentence", List(thisSentence.mentions.size.toString, y)) -> 1.0
    feats += FeatureKey("Arg candidate is protein", List(x.isProtein.toString, y)) -> 1.0
    //feats += FeatureKey("Arg isProtein and contains -", List((token.word.contains("-") && x.isProtein).toString, y)) -> 1.0
    // -----------------------------------------------------------------------------


    //                                 Syntax
    // -----------------------------------------------------------------------------
    feats += FeatureKey("Arg dependency between arg and source trigger head", List(thisSentence.deps.filter(e => (e.head == begin && e.mod == parentToken.begin)).toString, y)) -> 1.0
    feats += FeatureKey("Arg dependency between arg and source trigger mod", List(thisSentence.deps.filter(e => (e.mod == begin && e.head == parentToken.begin)).toString, y)) -> 1.0

    //feats += FeatureKey("Arg dependency existence between arg and source trigger", List(thisSentence.deps.filter(e => (e.head == begin && e.mod == parentToken.begin || e.mod == begin && e.head == parentToken.begin)).isEmpty.toString, y)) -> 1.0
    //feats += FeatureKey("Arg number of dependencies of parent", List(thisSentence.deps.filter(e => (e.head == parentToken.begin || e.mod == parentToken.begin)).size.toString, y)) -> 1.0
    //feats += FeatureKey("Arg number of dependencies of candidate", List(thisSentence.deps.filter(e => (e.head == token.begin || e.mod == token.begin)).size.toString, y)) -> 1.0
    // -----------------------------------------------------------------------------


    //                                 Other
    // ------------------------------------------------------------------------------
    //feats += FeatureKey("Candidate not protein and has same pos as parent", List(x.isProtein.toString, (token.pos == parentToken.pos).toString, y)) -> 1.0 // common occurance for theme arguments

    //if(begin >=1 && begin < thisSentence.tokens.size-1)
    //  feats += FeatureKey("silly pos trigram", List(thisSentence.tokens(begin-1).pos,token.pos,thisSentence.tokens(begin+1).pos,y)) -> 1.0

    if (begin >= 1) {
      feats += FeatureKey("Arg silly prior stem bigram", List(thisSentence.tokens(begin - 1).stem, token.stem, y)) -> 1.0
      //feats += FeatureKey("Arg silly prior word bigram", List(thisSentence.tokens(begin - 1).word, token.word, y)) -> 1.0
      feats += FeatureKey("Arg silly prior pos bigram", List(thisSentence.tokens(begin - 1).pos, token.pos, y)) -> 1.0
    } else {
      feats += FeatureKey("Arg silly prior stem bigram", List(y)) -> 1.0
      feats += FeatureKey("Arg silly prior pos bigram", List(y)) -> 1.0
    }

    // -----------------------------------------------------------------------------

    //                                Positional
    // -----------------------------------------------------------------------------
    feats+= FeatureKey("Arg absolute distance from candidate", List((Math.abs(token.index - x.parentIndex)).toString,y)) -> 1.0
    //feats+= FeatureKey("non-absolute distance from candidate", List((token.index - x.parentIndex).toString,y)) -> 1.0
    //feats += FeatureKey("Arg parent is left or right of candidate", List(Math.signum(x.begin - x.parentIndex).toString, y)) -> 1.0
    // -----------------------------------------------------------------------------


    feats.toMap
  }

  def myArgumentFeaturesNB(x: Candidate, y: Label): FeatureVector = {
    val doc = x.doc
    val begin = x.begin
    val end = x.end
    val thisSentence = doc.sentences(x.sentenceIndex)
    val token = thisSentence.tokens(begin)
    val parentToken = thisSentence.tokens(x.parentIndex)
    val parentCandidate = thisSentence.events(x.parentIndex)
    val heads = thisSentence.deps.filter(e => e.head == begin)
    val mods = thisSentence.deps.filter(e => e.mod == begin)

    val feats = new mutable.HashMap[FeatureKey, Double]

    //println(thisSentence.deps.filter(e=>(e.head == begin && e.mod == parentToken.begin || e.mod == begin && e.head == parentToken.begin)).toString

    //===========FEATURES===========

    //Label Bias
    feats += FeatureKey("Arg Label bias", List(y)) -> 1.0


    //feats += FeatureKey("Arg value of word", List(token.word, y)) -> 1.0
    //feats += FeatureKey("Arg pos of parentEvent", List(parentToken.pos,y)) -> 1.0
    //feats += FeatureKey("Arg Word Value", List(token.word, y)) -> 1.0
    //feats += FeatureKey("Arg Parent Value", List(parentToken.word, y)) -> 1.0
    //feats += FeatureKey("Arg pos of parent", List(parentToken.pos, y)) -> 1.0
    //feats += FeatureKey("Arg Abs Distance from Parent", List((Math.abs(token.index - x.parentIndex)).toString, y)) -> 1.0
    //feats += FeatureKey("Arg parent is left or right of candidate", List(Math.signum(x.begin - x.parentIndex).toString, y)) -> 1.0


    //Other Features
      feats += FeatureKey("Arg POS and isProtein", List(token.pos, x.isProtein.toString, y)) -> 1.0
      feats += FeatureKey("Arg pos of parent and candidate are equal", List((token.pos == parentToken.pos).toString, y)) -> 1.0 // helps both. generally helps argument extraction
      feats += FeatureKey("Arg Capital Letter Exists and isProtein", List(token.word.exists(_.isUpper).toString, x.isProtein.toString, y)) -> 1.0  // very STRONK BOY
      feats += FeatureKey("Arg Dependency between argument and Parent", List(thisSentence.deps.filter(e => (e.head == begin && e.mod == parentToken.begin || e.mod == begin && e.head == parentToken.begin)).toString, y)) -> 1.0
      feats += FeatureKey("Arg absolute distance from candidate < 40", List((Math.abs(token.index - x.parentIndex) < 40).toString,y)) -> 1.0
      feats += FeatureKey("Proteins in sentence", List((thisSentence.mentions.size > 0).toString, y)) -> 1.0
    if (begin >= 1) {
      feats += FeatureKey("Arg silly prior pos bigram", List(thisSentence.tokens(begin - 1).pos, token.pos, y)) -> 1.0
    }




    //feats += FeatureKey("Arg word of Parent and candidate", List(token.word, parentToken.word, y)) -> 1.0
    //feats += FeatureKey("Arg STEM of parent", List(parentToken.stem, y)) -> 1.0

    //feats += FeatureKey("Arg ")
    //feats += FeatureKey("Arg STEM of parent", List(parentToken.stem, y)) -> 1.0
    //feats += FeatureKey("Arg POS of word", List(token.pos, y)) -> 1.0
    //feats += FeatureKey("Arg isProtein", List(x.isProtein.toString, y)) -> 1.0

    /*
    F1 score
    ---------
    Average(excluding: Set(None)): 0.11949523564254441}
    Per class: Map(None -> 0.9375589837220347, Theme -> 0.12027342736004154, Cause -> 0.02173913043478261)
    feats += FeatureKey("Arg POS = NN, isProtein", List((token.pos == "NN" && x.isProtein).toString, y)) -> 1.0
    feats += FeatureKey("Arg pos of parent and candidate are equal", List((token.pos == parentToken.pos).toString, y)) -> 1.0 // helps both. generally helps argument extraction
    feats += FeatureKey("Arg Capital Letter Exists and isProtein", List(token.word.exists(_.isUpper).toString, x.isProtein.toString, y)) -> 1.0  // ability to classify theme goes down but cause goes up.
    feats += FeatureKey("Arg Dependency between argument and Parent", List(thisSentence.deps.filter(e => (e.head == begin && e.mod == parentToken.begin || e.mod == begin && e.head == parentToken.begin)).toString, y)) -> 1.09
*/

    feats.toMap
  }
}
