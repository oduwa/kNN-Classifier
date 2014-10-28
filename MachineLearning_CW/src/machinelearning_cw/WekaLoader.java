/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package machinelearning_cw;

import java.io.FileReader;
import java.io.IOException;
import weka.core.Instances;

/**
 *
 * @author Odie
 */
public class WekaLoader {

    public static Instances loadData(String fullPath) {
        Instances d = null;
        FileReader r;
        try {
            r = new FileReader(fullPath);
            d = new Instances(r);
            d.setClassIndex(d.numAttributes() - 1);
        } catch (IOException e) {
            System.out.println("Unable to load data on path " + fullPath + " Exception thrown =" + e);
            System.exit(0);
        }
        return d;
    }

}
