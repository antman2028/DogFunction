using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Linq;
using System.Collections.Generic;


namespace DogPredictionFunction.Models;

public class Image{
    private static readonly InferenceSession? _session = new InferenceSession("");
    public static List<List<List<int>>>? ImageMatrix;

    public Dictionary<string,float> createPredictions(){
        if (ImageMatrix == null){
            throw new ArgumentException("Image not submitted!");
        }
        var inputTensor = ConvertToTensor(ImageMatrix) ?? throw new Exception("Unable to create Tensor");

        var input = new List<NamedOnnxValue> {
            NamedOnnxValue.CreateFromTensor("input",inputTensor)
        };

        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(input);

        float[] output = results.First().AsTensor<float>().ToArray();

        return new Dictionary<string, float> {};

    }

    private Tensor<float> ConvertToTensor(List<List<List<int>>> imageData)
    {
        int height = imageData.Count;
        int width = imageData[0].Count;
        var tensor = new DenseTensor<float>(new int[] { 1, 3, height, width });

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                List<int> pixel = imageData[y][x];
                tensor[0, 0, y, x] = pixel[0] / 255f; // Red
                tensor[0, 1, y, x] = pixel[1] / 255f; // Green
                tensor[0, 2, y, x] = pixel[2] / 255f; // Blue
            }
        }

        return tensor;
    }
    
}