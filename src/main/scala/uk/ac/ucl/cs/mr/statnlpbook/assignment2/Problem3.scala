package uk.ac.ucl.cs.mr.statnlpbook.assignment2

/**
 * Created by Georgios on 06/11/2015.
 */
object Problem3Triggers {

  def main(args: Array[String]) {
    println("Trigger Extraction")
    val train_dir = "./data/assignment2/bionlp/train"
    val test_dir = "./data/assignment2/bionlp/test"

    // load train and dev data
    // read the specification of the method to load more/less data for debugging speedup
    val (trainDocs, devDocs) = BioNLP.getTrainDevDocuments(train_dir, 0.8, 500)
    // load test
    val testDocs = BioNLP.getTestDocuments(test_dir)
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
    val triggerTest = preprocess(getTestTriggerCandidates(testDocs))

    // show statistics for counts of true labels, useful for deciding on subsampling
    println("True label counts (trigger - train):")
    println(triggerTrain.unzip._2.groupBy(x => x).mapValues(_.length))
    println("True label counts (trigger - dev):")
    println(triggerDev.unzip._2.groupBy(x => x).mapValues(_.length))
    // get label set
    val triggerLabels = triggerTrain.map(_._2).toSet
//    println("THESE ARE THE LABELS " + triggerLabels)


    // define model
    //TODO: change the features function to explore different types of features
    val triggerModel = SimpleClassifier(triggerLabels, Features.myTriggerFeatures)

    // use training algorithm to get weights of model
    //TODO: change the trainer to explore different training algorithms
    val triggerWeights = PrecompiledTrainers.trainPerceptron(triggerTrain, triggerModel.feat, triggerModel.predict, 10)
    //val triggerWeights = PrecompiledTrainers.trainNB(triggerTrain,triggerModel.feat)

    // Outputting feature weights TODO remove this debugging code
    val range = 5
    val template = "Trigger: pos of word"
    val highest = false

    if (highest) {
      println(s"\nTAKING $range HIGHEST TRIGGER WEIGHTS FOR $template\n")
      for (x <- triggerWeights.filter(e => e._1.template == template).toList.sortBy(_._2).reverse.take(range).map(f => f._1.arguments -> f._2).zipWithIndex){
        println(x._2+1 + ": " + x._1.toString())
      }
    } else {
      println(s"\nTAKING $range LOWEST TRIGGER WEIGHTS FOR $template\n")
      for (x <- triggerWeights.filter(e => e._1.template == template).toList.sortBy(_._2).take(range).map(f => f._1.arguments -> f._2).zipWithIndex){
        println(x._2+1 + ": " + x._1.toString())
      }
    }

    // evaluate on dev
    // write to file
    // get predictions on test
    // get predictions on dev
    val (triggerDevPred, triggerDevGold) = triggerDev.map { case (trigger, gold) => (triggerModel.predict(trigger, triggerWeights), gold) }.unzip
    val triggerTestPred = triggerTest.map { case (trigger, dummy) => triggerModel.predict(trigger, triggerWeights) }
    // print evaluation results
    val triggerDevEval = Evaluation(triggerDevGold, triggerDevPred, Set("None"))
    println("Evaluation for trigger classification:")
    println(triggerDevEval.toString)

    ErrorAnalysis(triggerDev.unzip._1,triggerDevGold,triggerDevPred).showErrors(5)

    Evaluation.toFile(triggerTestPred, "./data/assignment2/out/simple_trigger_test.txt")
  }
}

object Problem3Arguments {
  def main (args: Array[String] ) {
    println("Arguments Extraction")
    val train_dir = "./data/assignment2/bionlp/train"
    val test_dir = "./data/assignment2/bionlp/test"

    // load train and dev data
    // read the specification of the method to load more/less data for debugging speedup
    val (trainDocs, devDocs) = BioNLP.getTrainDevDocuments(train_dir,0.8,500)
    // load test
    val testDocs = BioNLP.getTestDocuments(test_dir)
    // make tuples (Candidate,Gold)
    def preprocess(candidates: Seq[Candidate]) = candidates.map(e => e -> e.gold)

    // ================= Argument Classification =================

    // get candidates and make tuples with gold
    // no subsampling for dev/test!
    def getArgumentCandidates(docs:Seq[Document]) = docs.flatMap(_.argumentCandidates(0.008))
    def getTestArgumentCandidates(docs:Seq[Document]) = docs.flatMap(_.argumentCandidates())
    val argumentTrain =  preprocess(getArgumentCandidates(trainDocs))
    val argumentDev = preprocess(getTestArgumentCandidates(devDocs))
    val argumentTest = preprocess(getTestArgumentCandidates(testDocs))

    // show statistics for counts of true labels, useful for deciding on subsampling
    println("True label counts (argument - train):")
    println(argumentTrain.unzip._2.groupBy(x=>x).mapValues(_.length))
    println("True label counts (argument - dev):")
    println(argumentDev.unzip._2.groupBy(x=>x).mapValues(_.length))

    // get label set
    val argumentLabels = argumentTrain.map(_._2).toSet

    // define model
    //val argumentModel = SimpleClassifier(argumentLabels, Features.myArgumentFeaturesNB)
    val argumentModel = SimpleClassifier(argumentLabels, Features.myArgumentFeatures)

    //val argumentWeights = PrecompiledTrainers.trainNB(argumentTrain,argumentModel.feat)
    val argumentWeights = PrecompiledTrainers.trainPerceptron(argumentTrain,argumentModel.feat,argumentModel.predict,10)

    // Outputting feature weights
    val range = 5
    //val template = "Arg absolute distance from candidate", "Arg absolute distance from candidate"
    val highest = true

    val templates = List("Arg number of dependencies of candidate")

      //println(argumentWeights.filter(e => e._1.template == template).toList.sortBy(_._2).reverse.take(range))
    for (temp <- templates) {
      println("==================== " + temp + " ==========================")
      argumentWeights.toList.sortBy(_._2).reverse.filter(e => e._1.template == temp).map(e => println(e)) //.take(range))
    }


    //templates.map(f => argumentWeights.toList.sortBy(_._2).reverse.filter(e => e._1.template == f).map(e => println(e)))


    // get predictions on dev
    val (argumentDevPred, argumentDevGold) = argumentDev.map { case (arg, gold) => (argumentModel.predict(arg,argumentWeights), gold) }.unzip
    // evaluate on dev
    val argumentDevEval = Evaluation(argumentDevGold, argumentDevPred, Set("None"))
    println("Evaluation for argument classification:")
    println(argumentDevEval.toString)


    ErrorAnalysis(argumentDev.unzip._1,argumentDevGold,argumentDevPred).showErrors(5)

    // get predictions on test
    val argumentTestPred = argumentTest.map { case (arg, dummy) => argumentModel.predict(arg,argumentWeights) }
    // write to file
    Evaluation.toFile(argumentTestPred,"./data/assignment2/out/simple_argument_test.txt")
  }

}