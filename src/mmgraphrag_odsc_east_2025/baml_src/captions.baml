// Define Image model for the results of a LLaVA analysis
class ImageCaption {
  caption string @description(#"A concise description of the image."#)
}


// Create a function to extract the image from a base64 encoding of the image.
function ExtractImageCaption(img: image) -> ImageCaption {
  client OllamaGemma3_12b // OllamaLLaVA // OllamaPhi4
  prompt #"
    {{_.role("user")}}
    Write a caption for this image: {{img}}

    {{ ctx.output_format }}
  "#
}


test test_image_caption {
  functions [ExtractImageCaption]
  args {
    img {
      file "../images/e_COCO_val2014_000000000757.jpg"
    }
  }
}
