using Microsoft.ML;
using Microsoft.ML.Data;
using System;

namespace ClientPrediction
{
    // Class representing the input data schema
    public class EcommerceData
    {
        [LoadColumn(0)]
        public string EventTime { get; set; } // The timestamp of the event

        [LoadColumn(1)]
        public string EventType { get; set; } // The type of the event (e.g., view, purchase)

        [LoadColumn(2)]
        public float ProductId { get; set; } // Product identifier

        [LoadColumn(3)]
        public float CategoryId { get; set; } // Category identifier

        [LoadColumn(4)]
        public string CategoryCode { get; set; } // Category code

        [LoadColumn(5)]
        public string Brand { get; set; } // Product brand

        [LoadColumn(6)]
        public float Price { get; set; } // Product price

        [LoadColumn(7)]
        public float UserId { get; set; } // User identifier

        [LoadColumn(8)]
        public string UserSession { get; set; } // Session ID of the user
    }

    // Class for transformed data with the label column
    public class EcommerceTransformedData
    {
        public bool IsPurchase { get; set; } // Indicates if the event is a purchase
        public float ProductId { get; set; }
        public float CategoryId { get; set; }
        public float Price { get; set; }
        public float UserId { get; set; }
    }

    // Class for prediction results
    public class EcommercePrediction
    {
        [ColumnName("PredictedLabel")]
        public bool IsPurchase { get; set; } // Predicted value for the purchase
    }

    class Program
    {
        static void Main(string[] args)
        {
            // Create a new ML.NET context
            var context = new MLContext();

            // Specify the path to the CSV data file
            string dataPath = "C:\\Users\\2019-Oct.csv";

            // Load the data into an IDataView
            var dataView = context.Data.LoadFromTextFile<EcommerceData>(
                dataPath, separatorChar: ',', hasHeader: true);

            // Add a calculated column "IsPurchase" to indicate whether the event is a purchase
            var pipeline = context.Transforms.CustomMapping<EcommerceData, EcommerceTransformedData>(
                (input, output) =>
                {
                    output.IsPurchase = input.EventType == "purchase"; // Mark as purchase if the event type matches
                    output.ProductId = input.ProductId;
                    output.CategoryId = input.CategoryId;
                    output.Price = input.Price;
                    output.UserId = input.UserId;
                },
                contractName: "IsPurchaseMapping");

            // Transform the data
            var transformedData = pipeline.Fit(dataView).Transform(dataView);

            // Split the data into training and testing sets
            var split = context.Data.TrainTestSplit(transformedData, testFraction: 0.2);
            var trainData = split.TrainSet;
            var testData = split.TestSet;

            // Build the machine learning pipeline
            var trainingPipeline = context.Transforms.Concatenate("Features", "ProductId", "CategoryId", "Price", "UserId")
                .Append(context.Transforms.NormalizeMinMax("Features")) // Normalize features
                .Append(context.BinaryClassification.Trainers.FastTree(labelColumnName: "IsPurchase")); // Use FastTree for binary classification

            // Train the model
            var model = trainingPipeline.Fit(trainData);

            // Evaluate the model on the test dataset
            var predictions = model.Transform(testData);
            var metrics = context.BinaryClassification.Evaluate(predictions, labelColumnName: "IsPurchase");

            // Output evaluation metrics
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:P2}");

            // Check if the model's accuracy is acceptable
            if (metrics.Accuracy >= 0.7)
            {
                Console.WriteLine("The model performs well and can be used for predictions.");
            }
            else
            {
                Console.WriteLine("The model may not be accurate enough. Consider improving it.");
            }

            // Example data for prediction
            var sampleData = new EcommerceTransformedData()
            {
                ProductId = 44600062f,
                CategoryId = 2103807459595387724f,
                Price = 35.79f,
                UserId = 541312140f,
                IsPurchase = false // Not used for prediction
            };

            // Create a PredictionEngine for making predictions
            var predictionEngine = context.Model.CreatePredictionEngine<EcommerceTransformedData, EcommercePrediction>(model);

            // Perform the prediction
            var prediction = predictionEngine.Predict(sampleData);

            // Output the prediction result
            Console.WriteLine($"Predicted purchase: {prediction.IsPurchase}");
        }
    }
}
