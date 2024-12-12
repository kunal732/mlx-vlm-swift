# MLX-VLM-Swift
This is a stand alone package that lets you quickly load and make inferences on Vision Language Models. You just provide an image(s), a prompt & receieve a textual output. 

## Attribution
This repository is a standalone fork of the immense work by **David Kowski** and the MLX team in porting MLX-VLM from Python to Swift, originally showcased in the expansive [mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples/) repository. I created this repo solely as a way to streamline maintenance & facilitate adding support for the latest MLX vision-capable models more easily.

## Installation

This project is distributed as a Swift package. You can add it to your Xcode project as follows:

1. In Xcode, go to **File > Add Packages...**
2. Paste the URL of this repository into the search bar:

## Usage

Below is a barebones code snippet showing how you can load a vision-language model and perform inference with a prompt and an image:

```swift
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM
import Tokenizers
import Foundation

// 1. Provide the model name (Hugging Face repository or local directory)
let modelName = "mlx-community/paligemma2-3b-ft-docci-448-8bit"

// 2. Specify your input prompt and an image path
let prompt = "Describe the image in detail."
let imagePath = "/path/to/your/image.png"

// 3. Asynchronously load the model
let modelFactory = VLMModelFactory.shared
let modelConfiguration = modelFactory.configuration(id: modelName)

Task {
 do {
     // Load the model container
     let modelContainer = try await modelFactory.loadContainer(configuration: modelConfiguration)

     // 4. Prepare your input
     let input = UserInput(
         prompt: prompt,
         images: [.url(URL(fileURLWithPath: imagePath))]
     )

     // 5. Perform inference
     let result = try await modelContainer.perform { context in
         // Process input
         let processedInput = try await context.processor.prepare(input: input)
         
         // Generate text from the prompt and image
         let output = try MLXLMCommon.generate(
             input: processedInput,
             parameters: GenerateParameters(temperature: 0.7),
             context: context
         )

         return output
     }

     // 6. Print the result
     print("Model Output:")
     print(result.summary())

 } catch {
     print("Error: \(error)")
 }

}
```
This snippet:
- **Loads a model**: Given a Hugging Face model name, it downloads and configures the model.
- **Provides a prompt and an image as input**: The input prompt and image are processed for the model.
- **Runs inference and prints the output text**: Generates a textual response based on the input.
- **No additional classes or tasks beyond the standard MLX interfaces are required**.

