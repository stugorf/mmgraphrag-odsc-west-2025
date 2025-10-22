// Define models for visual object detection and audio analysis

class AudioObject {
  name string @description(#"Name of the detected sound"#)
  start_time float? @description(#"Start time of the sound in seconds"#)
  duration float? @description(#"Duration of the sound in seconds"#)
  count int? @description(#"How many of this sound type are detected in the audio file"#)
  location string? @description(#"Estimated location (left, center, right) based on stereo audio"#)
  travel_of_direction string? @description(#"Estimated direction (going left, going right, going forward, going backward) based on stereo audio, doppler effect, volume, etc."#)
  speed float? @description(#"Estimated speed (in meters per second) based on stereo audio"#)
  explanation string? @description(#"Explanation of why you think this is the name of the sound"#)

}


class AudioAnalysis {
  audio_objects AudioObject[] @description(#"List of detected audio objects, there may be multiple objects in the same time frame or overlapping."#)
  temporal_relationships string[] @description(#"Sequence of sounds and their relationships"#)
}

class Speaker {
  name string @description(#"Name of the speaker; if not known, use alias like 'Speaker_1' or 'Speaker_2' etc."#)
  position string? @description(#"Estimated direction (left, center, right)"#)
}

class Sentiment {
  sentiment string @description(#"Sentiment of the audio"#)
  confidence float @description(#"Confidence in the sentiment"#)
}

class Emotion {
  emotion string @description(#"Emotion of the audio"#)
  confidence float @description(#"Confidence in the emotion"#)
}


function AnalyzeAudio(audioFile: audio) -> AudioAnalysis {
  client ChatGPTAudioPreview
  prompt #"
    Listen to the provided audio file and analyze its content thoroughly. Provide a detailed analysis. 
    The audio may contain multiple objects, so make sure to detect all of them. You may need to use doppler effect, volume, etc. to determine the number of objects and their direction. 
    
    {{ ctx.output_format }}

    {{ _.role("user") }} {{ audioFile }}
  "#
}


test test_audio_analysis {
  functions [AnalyzeAudio]
  args {
    audioFile {
      file "../audio/police_cars.mp3"
    }
  }
}