from dubstep_voz import generate_voice, create_dubstep_track, TEXT_TO_SPEECH

if __name__ == '__main__':
    print('Generating voice...')
    voice = generate_voice(TEXT_TO_SPEECH)
    if voice is None:
        print('Voice generation failed; exiting')
        exit(1)
    print('Creating track...')
    track = create_dubstep_track(voice)
    print('Exporting mix.wav and mix.mp3...')
    try:
        track.export('mix.wav', format='wav')
        track.export('mix.mp3', format='mp3')
        print('Export complete.')
    except Exception as e:
        print('Export failed:', e)
