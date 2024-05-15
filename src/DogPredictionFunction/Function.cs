using Amazon.Lambda.Core;
using DogPredictionFunction.Models;
using System.Text.Json;


[assembly: LambdaSerializer(typeof(Amazon.Lambda.Serialization.SystemTextJson.DefaultLambdaJsonSerializer))]

namespace DogPredictionFunction;

public class Function
{
    
    public string FunctionHandler(string input, ILambdaContext context)
    {
        Image petImage = JsonSerializer.Deserialize<Image>(input) ?? throw new Exception("Image unable to deserialize");
        Dictionary<string,float> breedPercentages = petImage.createPredictions();
        return JsonSerializer.Serialize(breedPercentages);
    }
}
