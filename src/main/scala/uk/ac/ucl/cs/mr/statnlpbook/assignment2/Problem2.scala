package uk.ac.ucl.cs.mr.statnlpbook.assignment2

import scala.collection.mutable

/**
 * Created by Georgios on 11/11/2015.
 */
object Problem2 {


  /**
   * Train a linear model using the average perceptron algorithm.
   * @param instances the training instances.
   * @param feat a joint feature function.
   * @param predict a prediction function that maps inputs to outputs using the given weights.
   * @param iterations number of iterations.
   * @param learningRate
   * @tparam X type of input.
   * @tparam Y type of output.
   * @return a linear model trained using the perceptron algorithm.
   */
  def trainAvgPerceptron[X, Y](instances: Seq[(X, Y)],
                               feat: (X, Y) => FeatureVector,
                               predict: (X, Weights) => Y,
                               iterations: Int = 2,
                               learningRate: Double = 1.0): Weights = {

    val weights = new mutable.HashMap[FeatureKey, Double]() withDefaultValue 0.0
    val avgWeights = new mutable.HashMap[FeatureKey, Double]() withDefaultValue 0.0
    val lastUpdate = new mutable.HashMap[FeatureKey, Double]() withDefaultValue 0.0 // last update of each feature (iteration)
    var count = 0.0
    for (i <- 0 to iterations){
      for ((e, g) <- instances) {
        val p = predict(e,weights)
        count+=1.0 // weights have been used once more
        if(p != g){ // if prediction is wrong - not gold
          val goldFeats = feat(e,g)
          val predFeats = feat(e,p)
          for ((k, v) <- goldFeats ++ predFeats) { // iterate over each changed feature
            avgWeights(k) += weights(k) * (count - lastUpdate(k)) // current weight times duration of it
            lastUpdate(k) = count // about to be updated so last update is current count
          }
          addInPlace(goldFeats, weights, 1.0) // increase weighting for gold
          addInPlace(predFeats, weights, -1.0) // decrease weighting for incorrect prediction
        }
      }
    }
    for ((k,v) <- weights){
      avgWeights(k) += v * (count - lastUpdate(k)) // Final update to ensure all weights are included up to final count
    }
    for ((k,_) <- avgWeights) avgWeights(k) /= count // divide each value by number summed over to give average
    avgWeights
  }
// alternative is to store last update time of each feature as a separate map

  /**
   * Run this code to evaluate your implementation of your avereaged perceptron algorithm trainer
   * Results should be similar to the precompiled trainer
   * @param args
   */
  def main (args: Array[String] ) {

    val train_dir = "./data/assignment2/bionlp/train"

    // load train and dev data
    // read the specification of the method to load more/less data for debugging speedup
    val (trainDocs, devDocs) = BioNLP.getTrainDevDocuments(train_dir, 0.8, 100)
    // make tuples (Candidate,Gold)
    def preprocess(candidates: Seq[Candidate]) = candidates.map(e => e -> e.gold)

    // ================= Trigger Classification =================

    // get candidates and make tuples with gold
    // read the specifications of the method for different subsampling thresholds
    // no subsampling for dev/test!
    def getTriggerCandidates(docs: Seq[Document]) = docs.flatMap(_.triggerCandidates(0.02))
    def getTestTriggerCandidates(docs: Seq[Document]) = docs.flatMap(_.triggerCandidates())
    val triggerTrain = preprocess(getTriggerCandidates(trainDocs))
    val triggerDev = preprocess(getTestTriggerCandidates(devDocs))

    // get label set
    val triggerLabels = triggerTrain.map(_._2).toSet

    // define model
    val triggerModel = SimpleClassifier(triggerLabels, defaultTriggerFeatures)

    val myWeights = trainAvgPerceptron(triggerTrain, triggerModel.feat, triggerModel.predict, 1)
    val precompiledWeights = PrecompiledTrainers.trainAvgPerceptron(triggerTrain, triggerModel.feat, triggerModel.predict, 1)

    // get predictions on dev
    val (myPred, gold) = triggerDev.map { case (trigger, gold) => (triggerModel.predict(trigger, myWeights), gold) }.unzip
    val (precompiledPred, _) = triggerDev.map { case (trigger, gold) => (triggerModel.predict(trigger, precompiledWeights), gold) }.unzip

    // evaluate models (dev)
    println("Evaluation - my trainer:")
    println(Evaluation(gold, myPred, Set("None")).toString)
    println("Evaluation - precompiled trainer:")
    println(Evaluation(gold, precompiledPred, Set("None")).toString)
  }

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


}
