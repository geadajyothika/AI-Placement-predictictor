import weka.core.Instance;
import weka.core.Instances;
import weka.core.DenseInstance;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;

import java.util.Scanner;

public class PredictPlacement {

    public static void main(String[] args) throws Exception {

        DataSource source = new DataSource("dataset.arff");
        Instances data = source.getDataSet();

        data.setClassIndex(data.numAttributes() - 1);

        J48 tree = new J48();
        tree.buildClassifier(data);

        Scanner sc = new Scanner(System.in);

        System.out.println("Enter CGPA:");
        double cgpa = sc.nextDouble();

        System.out.println("Enter number of projects:");
        double projects = sc.nextDouble();

        System.out.println("Programming level (0=low,1=medium,2=high):");
        int programming = sc.nextInt();

        System.out.println("Internship (0=no,1=yes):");
        int internship = sc.nextInt();

        Instance newStudent = new DenseInstance(5);
        newStudent.setDataset(data);

        newStudent.setValue(0, cgpa);
        newStudent.setValue(1, projects);
        newStudent.setValue(2, programming);
        newStudent.setValue(3, internship);

        double result = tree.classifyInstance(newStudent);

        System.out.println("Prediction: " + data.classAttribute().value((int) result));
    }
}