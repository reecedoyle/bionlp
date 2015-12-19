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
    val parentEvent = thisSentence.events(x.parentIndex)
    val parentToken = thisSentence.tokens(parentEvent.begin)
    val heads = thisSentence.deps.filter(e => e.head == begin)
    val mods = thisSentence.deps.filter(e => e.mod == begin)

    val feats = new mutable.HashMap[FeatureKey, Double]

    feats += FeatureKey("Arg Label bias", List(y)) -> 1.0

    //                                 Lexical
    // -----------------------------------------------------------------------------
    //Part-Of-Speech
    feats += FeatureKey("Arg pos of parent and candidate are equal", List((token.pos == parentToken.pos).toString, y)) -> 1.0
    feats += FeatureKey("Arg POS of candidate and parent", List(token.pos, parentToken.pos, y)) -> 1.0
    //feats += FeatureKey("Arg POS", List(token.pos, y)) -> 1.0
    //feats += FeatureKey("Arg Parent POS", List(parentToken.pos, y)) -> 1.0

    //Words
    //feats += FeatureKey("Arg Word", List(token.word, y)) -> 1.0
    //feats += FeatureKey("Arg Parent Word", List(token.word, y)) -> 1.0
    //feats += FeatureKey("Arg Word of candidate and parent", List(token.word, parentToken.word, y)) -> 1.0
    //feats += FeatureKey("word contains reg", List(token.word.toLowerCase.contains("reg").toString,y)) -> 1.0


    //Stems
    feats += FeatureKey("Arg Stem of candidate and parent", List(parentToken.stem, token.stem, y)) -> 1.0
    //feats += FeatureKey("Arg Stem of Parent", List(parentToken.stem, y)) -> 1.0
    //feats += FeatureKey("Arg Stem", List(token.stem, y)) -> 1.0

    feats += FeatureKey("Arg Stem of Parent and isProtein", List(parentToken.stem, x.isProtein.toString, y)) -> 1.0

    //Miscellaneous and Combinational
    feats += FeatureKey("Arg capitalisation of candidate and is protein", List(token.word.exists(_.isUpper).toString, x.isProtein.toString, y)) -> 1.0
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
    //Mcclosky Dependencies
    feats += FeatureKey("Arg dependency between arg and source trigger head", List(thisSentence.deps.filter(e => (e.head == begin && e.mod == parentToken.begin)).toString, y)) -> 1.0
    feats += FeatureKey("Arg dependency between arg and source trigger mod", List(thisSentence.deps.filter(e => (e.mod == begin && e.head == parentToken.begin)).toString, y)) -> 1.0
    //feats += FeatureKey("Arg dependency existence between arg and source trigger", List(thisSentence.deps.filter(e => (e.head == begin && e.mod == parentToken.begin || e.mod == begin && e.head == parentToken.begin)).isEmpty.toString, y)) -> 1.0
    //feats += FeatureKey("Arg number of dependencies of parent", List(thisSentence.deps.filter(e => (e.head == parentToken.begin || e.mod == parentToken.begin)).size.toString, y)) -> 1.0
    //feats += FeatureKey("Arg number of dependencies of candidate", List(thisSentence.deps.filter(e => (e.head == token.begin || e.mod == token.begin)).size.toString, y)) -> 1.0
    // -----------------------------------------------------------------------------


    //                                 Prior Word
    // ------------------------------------------------------------------------------
    //if(begin >=1 && begin < thisSentence.tokens.size-1)
    //  feats += FeatureKey("silly pos trigram", List(thisSentence.tokens(begin-1).pos,token.pos,thisSentence.tokens(begin+1).pos,y)) -> 1.0

    if (begin >= 1) {
      feats += FeatureKey("Arg Prior Stem", List(thisSentence.tokens(begin - 1).stem, token.stem, y)) -> 1.0
      //feats += FeatureKey("Arg Prior Word"), List(thisSentence.tokens(begin - 1).word, token.word, y)) -> 1.0
      feats += FeatureKey("Arg Prior POS", List(thisSentence.tokens(begin - 1).pos, token.pos, y)) -> 1.0
    } else {
      feats += FeatureKey("Arg Prior Stem", List(y)) -> 1.0
      //feats += FeatureKey("Arg Prior Word", List(y)) -> 1.0
      feats += FeatureKey("Arg Prior POS", List(y)) -> 1.0
    }
    // -----------------------------------------------------------------------------

    //                                Positional
    // -----------------------------------------------------------------------------
    feats+= FeatureKey("Arg Absolute distance between candidate and parent", List((Math.abs(token.index - x.parentIndex)).toString,y)) -> 1.0
    //feat s+= FeatureKey("Arg distance between candidate and parent", List((token.index - x.parentIndex).toString,y)) -> 1.0
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

    //Label Bias
    feats += FeatureKey("Arg Label bias", List(y)) -> 1.0

    //                                 Lexical
    // -----------------------------------------------------------------------------
    feats += FeatureKey("Arg pos of parent and candidate are equal", List((token.pos == parentToken.pos).toString, y)) -> 1.0
    feats += FeatureKey("Arg POS and isProtein", List(token.pos, x.isProtein.toString, y)) -> 1.0
    feats += FeatureKey("Arg Capital Letter Exists and isProtein", List(token.word.exists(_.isUpper).toString, x.isProtein.toString, y)) -> 1.0
    // -----------------------------------------------------------------------------

    //                                Entity
    // -----------------------------------------------------------------------------
    feats += FeatureKey("Arg number of Proteins in sentence", List((thisSentence.mentions.size > 0).toString, y)) -> 1.0
    // -----------------------------------------------------------------------------

    //                                Syntax
    // -----------------------------------------------------------------------------
    feats += FeatureKey("Arg Dependency between argument and Parent head", List(thisSentence.deps.filter(e => (e.head == begin && e.mod == parentToken.begin)).toString, y)) -> 1.0
    feats += FeatureKey("Arg Depencency between argument and Parent mod", List(thisSentence.deps.filter(e => (e.mod == begin && e.head == parentToken.begin)).toString, y)) -> 1.0
    // -----------------------------------------------------------------------------

    //                              Prior Word
    // -----------------------------------------------------------------------------
    if (begin >= 1) {
      feats += FeatureKey("Arg Prior POS", List(thisSentence.tokens(begin - 1).pos, token.pos, y)) -> 1.0
    } else {
      feats += FeatureKey("Arg Prior POS", List(y)) -> 1.0

    }
    // -----------------------------------------------------------------------------

    //                              Positional
    // -----------------------------------------------------------------------------
    feats += FeatureKey("Arg absolute distance from candidate < 40", List((Math.abs(token.index - x.parentIndex) < 40).toString,y)) -> 1.0
    // -----------------------------------------------------------------------------

    feats.toMap
  }
}
