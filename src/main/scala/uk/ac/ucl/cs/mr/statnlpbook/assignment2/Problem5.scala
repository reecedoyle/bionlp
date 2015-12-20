package uk.ac.ucl.cs.mr.statnlpbook.assignment2

import scala.collection.mutable

/**
 * Created by Georgios on 30/10/2015.
 */

object Problem5{
  //TODO remove these
  val predMap = new mutable.HashMap[(Label,Label), Int]() withDefaultValue 0
  val argPredMap = new mutable.HashMap[(Label,Label), Int]() withDefaultValue 0
  var training = true

  def main (args: Array[String]) {
    println("Joint Extraction")

    val train_dir = "./data/assignment2/bionlp/train"
    val test_dir = "./data/assignment2/bionlp/test"

    // load train and dev data
    // read the specification of the method to load more/less data for debugging speedup
    val (trainDocs, devDocs) = BioNLP.getTrainDevDocuments(train_dir,0.8,500)
    // load test
    val testDocs = BioNLP.getTestDocuments(test_dir)
    // make tuples (Candidate,Gold)
    def preprocess(candidates: Seq[Candidate]) = candidates.map(e => e -> (e.gold,e.arguments.map(_.gold)))

    // ================= Joint Classification =================
    // get candidates and make tuples with gold
    // read the specifications of the method for different subsampling thresholds
    // no subsampling for dev/test!
    def getJointCandidates(docs: Seq[Document]) = docs.flatMap(_.jointCandidates(0.02,0.4))
    def getTestJointCandidates(docs: Seq[Document]) = docs.flatMap(_.jointCandidates())
    val jointTrain = preprocess(getJointCandidates(trainDocs))
    val jointDev = preprocess(getTestJointCandidates(devDocs))
    val jointTest = preprocess(getTestJointCandidates(testDocs))

    // show statistics for counts of true labels, useful for deciding on subsampling
    println("True label counts (trigger - train):")
    println(jointTrain.unzip._2.unzip._1.groupBy(x=>x).mapValues(_.length))
    println("True label counts (trigger - dev):")
    println(jointDev.unzip._2.unzip._1.groupBy(x=>x).mapValues(_.length))
    println("True label counts (argument - train):")
    println(jointTrain.unzip._2.unzip._2.flatten.groupBy(x=>x).mapValues(_.length))
    println("True label counts (argument - dev):")
    println(jointDev.unzip._2.unzip._2.flatten.groupBy(x=>x).mapValues(_.length))


    // get label sets
    val triggerLabels = jointTrain.map(_._2._1).toSet
    val argumentLabels = jointTrain.flatMap(_._2._2).toSet

    // define model
    val jointModel = JointUnconstrainedClassifier(triggerLabels,argumentLabels,Features.myTriggerFeatures,Features.myArgumentFeatures)
    //val jointModel = JointConstrainedClassifier(triggerLabels,argumentLabels,Features.myTriggerFeatures,Features.myArgumentFeatures)

    // use training algorithm to get weights of model
    val jointWeights = PrecompiledTrainers.trainPerceptron(jointTrain,jointModel.feat,jointModel.predict,10)

    training = false // TODO remove
    // get predictions on dev
    val jointDevPred = jointDev.unzip._1.map { case e => jointModel.predict(e,jointWeights) }
    val jointDevGold = jointDev.unzip._2
    training = true // TODO remove
    // Triggers (dev)
    val triggerDevPred = jointDevPred.unzip._1
    val triggerDevGold = jointDevGold.unzip._1
    val triggerDevEval = Evaluation(triggerDevGold,triggerDevPred,Set("None"))
    println("Evaluation for trigger classification:")
    println(triggerDevEval.toString)

    // Arguments (dev)
    val argumentDevPred = jointDevPred.unzip._2.flatten
    val argumentDevGold = jointDevGold.unzip._2.flatten
    val argumentDevEval = Evaluation(argumentDevGold,argumentDevPred,Set("None"))
    println("Evaluation for argument classification:")
    println(argumentDevEval.toString)

    // get predictions on test
    val jointTestPred = jointTest.unzip._1.map { case e => jointModel.predict(e,jointWeights) }
    // Triggers (test)
    val triggerTestPred = jointTestPred.unzip._1
    // write to file
    Evaluation.toFile(triggerTestPred,"./data/assignment2/out/joint_trigger_test.txt")
    // Arguments (test)
    val argumentTestPred = jointTestPred.unzip._2.flatten
    // write to file
    Evaluation.toFile(argumentTestPred,"./data/assignment2/out/joint_argument_test.txt")

    // TODO remove debugging below
    println("Trigger Predictions:")
    printf("%20s\t", "Gold | Pred ->")
    for (pred <- triggerLabels){
      printf("%20s\t", pred)
    }
    print("\n")
    for (gold <- triggerLabels){
      printf("%20s\t", gold)
      for (pred <- triggerLabels){
        printf("%20d\t", predMap(pred,gold))
      }
      print("\n")
    }

    println("\nArgument Predictions:")
    printf("%15s\t", "Gold | Pred ->")
    for (pred <- argumentLabels){
      printf("%15s\t", pred)
    }
    print("\n")
    for (gold <- argumentLabels){
      printf("%15s\t", gold)
      for (pred <- argumentLabels){
        printf("%15d\t", argPredMap(pred,gold))
      }
      print("\n")
    }
  }

}

/**
 * A joint event classifier (both triggers and arguments).
 * It predicts the structured event.
 * It's predict method should only produce the best solution that respects the constraints on the event structure.
 * @param triggerLabels
 * @param argumentLabels
 * @param triggerFeature
 * @param argumentFeature
 */
case class JointConstrainedClassifier(triggerLabels:Set[Label],
                                      argumentLabels:Set[Label],
                                      triggerFeature:(Candidate,Label)=>FeatureVector,
                                      argumentFeature:(Candidate,Label)=>FeatureVector
                                       ) extends JointModel {
  def predict(x: Candidate, weights: Weights) = {
    // following method is the argmax steps that concern the arguments
    def argumentsArgmax(labels: Set[Label], arguments: Seq[Candidate], weights: Weights, feat:(Candidate,Label)=>FeatureVector, tlabel: Label) = {
      val scores = for (arg<-arguments) yield labels.toSeq.map(y => y -> dot(feat(arg, y), weights)).toMap withDefaultValue 0.0 //e.g. Map(None -> 5.0, Theme -> -8.0, Cause -> 0.0)
      if(tlabel != "None"){ // match all times of regulation
        var maxArgScores = (0.0, Seq[Label]()) // stores max score and related labels
        var firstArg = true
        for (i <- arguments.indices){ // set each arg to theme in turn, then calculate score & see if better than current max
          val maxScoresLower = scores.zipWithIndex.filter(_._2<i).map(_._1.maxBy(_._2)) // Seq[(Label, Double)] = max label & score for each other argument
          val maxScoresHigher = scores.zipWithIndex.filter(_._2>i).map(_._1.maxBy(_._2)) // Seq[(Label, Double)]
          val totalScore = scores(i)("Theme") + maxScoresLower.map(_._2).sum + maxScoresHigher.map(_._2).sum // Double - total score for these arguments
          val predictedArgLabels = maxScoresLower.map(_._1)++Seq("Theme")++maxScoresHigher.map(_._1)

          if(maxArgScores._1 < totalScore || firstArg){ // if the score when this arg being Theme is better
            maxArgScores = (totalScore, predictedArgLabels)
            firstArg = false
          }
        }
        // return best argScores element
        maxArgScores
      }
      else{ // trigger label is none
        val maxScores = scores.map(_.maxBy(_._2)) // Seq[(Label, Double)]
        (maxScores.map(_._2).sum, maxScores.map(_._1)) // (Double, Seq[Label])
      }
    }
    def argmax() = {
      val triggerScores = triggerLabels.toSeq.map(y => y -> dot(triggerFeature(x, y), weights)).toMap withDefaultValue 0.0
      val argScores = new mutable.HashMap[Label, (Double, Seq[Label])]// stores max scores of all args & their labels for each trigger label
      for (tlabel <- triggerLabels){
        var currentArgLabels = argumentLabels
        if(tlabel == "None"){
          currentArgLabels = currentArgLabels - "Theme"
        }
        if(!tlabel.contains("egulation")){ // is not a regulation event
          currentArgLabels = currentArgLabels - "Cause"
        }
        argScores(tlabel) = argumentsArgmax(currentArgLabels,x.arguments,weights,argumentFeature, tlabel) // return best score & labels (Double, Seq[Label])
      }
      val maxTrigLabel = triggerScores.maxBy(t => t._2 + argScores(t._1)._1)._1
      if(!Problem5.training){
        Problem5.predMap(maxTrigLabel,x.gold) += 1
        for(i <- x.arguments.indices) Problem5.argPredMap(argScores(maxTrigLabel)._2(i), x.arguments(i).gold)+=1
      }
      (maxTrigLabel,argScores(maxTrigLabel)._2)
    }

    argmax()
  }

}

/**
 * A joint event classifier (both triggers and arguments).
 * It predicts the structured event.
 * It treats triggers and arguments independently, i.e. it ignores any solution constraints.
 * @param triggerLabels
 * @param argumentLabels
 * @param triggerFeature
 * @param argumentFeature
 */
case class JointUnconstrainedClassifier(triggerLabels:Set[Label],
                                        argumentLabels:Set[Label],
                                        triggerFeature:(Candidate,Label)=>FeatureVector,
                                        argumentFeature:(Candidate,Label)=>FeatureVector
                                         ) extends JointModel{
  /**
   * Constraint 1: if e=None, all a=None
   * Constraint 2: if e!=None, at least one a=Theme
   * Constraint 3: only e=Regulation can have a=Cause
   * @param x
   * @param weights
   * @return
   */
  def predict(x: Candidate, weights: Weights) = {
    def argmax(labels: Set[Label], x: Candidate, weights: Weights, feat:(Candidate,Label)=>FeatureVector) = {
      val scores = labels.toSeq.map(y => y -> dot(feat(x, y), weights)).toMap withDefaultValue 0.0
      scores.maxBy(_._2)._1
    }
    val bestTrigger = argmax(triggerLabels,x,weights,triggerFeature)
    val bestArguments = for (arg<-x.arguments) yield argmax(argumentLabels,arg,weights,argumentFeature)
    if(!Problem5.training){
      Problem5.predMap(bestTrigger,x.gold) += 1
      for(i <- x.arguments.indices) Problem5.argPredMap(bestArguments(i), x.arguments(i).gold)+=1
    }
    (bestTrigger,bestArguments)
  }

}

trait JointModel extends Model[Candidate,StructuredLabels]{
  def triggerFeature:(Candidate,Label)=>FeatureVector
  def argumentFeature:(Candidate,Label)=>FeatureVector
  def feat(x: Candidate, y: StructuredLabels): FeatureVector ={
    val f = new mutable.HashMap[FeatureKey, Double] withDefaultValue 0.0
    addInPlace(triggerFeature(x,y._1),f,1)
    for ((a,label)<- x.arguments zip y._2){
      addInPlace(argumentFeature(a,label),f,1)
    }
    f
  }
}


