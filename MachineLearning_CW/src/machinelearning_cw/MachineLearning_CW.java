/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package machinelearning_cw;

import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author 100024721
 */
public class MachineLearning_CW {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        // TODO code application logic here
        
        Instances train = WekaLoader.loadData("PitcherTrain.arff");
        Instances test = WekaLoader.loadData("PitcherTest.arff");
        
        /*
        BasicKNN basicKNN = new BasicKNN();
        basicKNN.buildClassifier(train);
        basicKNN.classifyInstance(test.firstInstance());
         */
        
        ///*
        //System.out.println(train + "\n\n\n");
        KNN knn = new KNN();
        //knn.setAutoDetermineK(true);
        knn.setUseWeightedVoting(true);
        knn.setUseAcceleratedNNSearch(true);
        knn.buildClassifier(train);
        //knn.testEstimateK();
        System.out.println("DECISION: " + knn.classifyInstance(test.get(0)));
        //*/
        for(Instance instance : train){
           // System.out.println(knn.distance(test.firstInstance(), instance));;
        }
        
        //System.out.println(knn.findNClosestNeighbourWithOrchards(test.get(0), train, 3));
        
        //System.out.println(knn.orchardsAlgorithm(test.get(1), train));;
    }
    
}
