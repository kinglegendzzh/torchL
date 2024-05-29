#include <iostream>
#include <aubio/aubio.h>

void detectChords(const char* filepath) {
    uint_t samplerate = 0;
    uint_t hop_size = 512;
    fvec_t* in = new_fvec(hop_size); // input buffer
    aubio_source_t* source = new_aubio_source(filepath, samplerate, hop_size);

    if (!source) {
        std::cerr << "Error opening source file " << filepath << std::endl;
        return;
    }

    samplerate = aubio_source_get_samplerate(source);
    uint_t total_frames = aubio_source_get_duration(source);
    std::cout << "Samplerate: " << samplerate << ", Total frames: " << total_frames << std::endl;

    // Create aubio pitch object
    aubio_pitch_t* pitch = new_aubio_pitch("default", hop_size, hop_size, samplerate);
    aubio_pitch_set_unit(pitch, "Hz");
    aubio_pitch_set_silence(pitch, -40);

    // Create aubio tempo object
    aubio_tempo_t* tempo = new_aubio_tempo("default", hop_size, hop_size, samplerate);

    // Process the audio file
    uint_t read = hop_size;
    while (read == hop_size) {
        aubio_source_do(source, in, &read);

        // Pitch detection
        fvec_t* out_pitch = new_fvec(1);
        aubio_pitch_do(pitch, in, out_pitch);
        std::cout << "Pitch: " << fvec_get_sample(out_pitch, 0) << " Hz" << std::endl;
        del_fvec(out_pitch);

        // Tempo detection
        fvec_t* out_tempo = new_fvec(2);
        aubio_tempo_do(tempo, in, out_tempo);
        if (fvec_get_sample(out_tempo, 0) != 0) {
            std::cout << "Beat at " << fvec_get_sample(out_tempo, 1) << " frames" << std::endl;
        }
        del_fvec(out_tempo);
    }

    // Clean up
    del_aubio_pitch(pitch);
    del_aubio_tempo(tempo);
    del_aubio_source(source);
    del_fvec(in);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <audio file>" << std::endl;
        return 1;
    }
    
    const char* filepath = argv[1];
    detectChords(filepath);

    // Wait for user input before closing
    std::cout << "Press Enter to exit...";
    std::cin.get();

    return 0;
}
