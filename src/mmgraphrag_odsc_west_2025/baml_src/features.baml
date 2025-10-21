// Define Image model for the results of a LLaVA analysis

class ImageObject {
  name string @description(#"A single word name for the object."#)
  colors string[] @description(#"A list of the top three predominant colors found in just the object."#)
  bounding_box float[] @description(#"The bounding box for the object. For example [194.26, 236.69, 21.22, 14.05]"#)

}

class ImageFeatures {
  colors string[] @description(#"A list of the top three predominant colors found in the entire image."#)
  sharpness string @description(#"A single word description of the image's sharpness"#)
  complexity string @description(#"A single word description of the image's complexity"#)
  textures string[] @description(#"A list of single words describing the textures in the image"#)
}

class ImageAnalysis {
  description ImageCaption
  features ImageFeatures
  objects ImageObject[]?
}

function ExtractImageFeatures(img: image) -> ImageFeatures {
    client OllamaLLaVA
    prompt #"
      {{_.role("user")}}
      Extract features in this image: {{img}}

      {{ ctx.output_format }}
    "#
}

function ExtractImageObjects(img: image) -> ImageObject[] {
    client OllamaLLaVA
    prompt #"
      {{_.role("user")}}
      Extract objects in this image: {{img}}

      {{ ctx.output_format }}
    "#
}

function AnalyzeImage(img: image) -> ImageAnalysis {
  client OllamaLLaVA
    prompt #"
      {{_.role("user")}}
      Extract the caption and features in this image: {{img}}

      {{ ctx.output_format }}
    "#
}


test test_image_features {
  functions [ExtractImageFeatures]
  args {
    img {
      file "../images/e_COCO_val2014_000000000757.jpg"
   }
  }
}

test test_image_objects {
  functions [ExtractImageObjects]
  args {
    img {
      file "../images/e_COCO_val2014_000000000757.jpg"
   }
  }
}

test test_image_analysis {
  functions [AnalyzeImage]
  args {
    img {
      file "../images/e_COCO_val2014_000000000757.jpg"
   }
  }
}