# vhs tape for the Skylar COBOL demo GIF.
# Render:  vhs assets/skylar-cobol-demo.tape   (needs vhs + ttyd + ffmpeg)
# Output 100% reale: il modello locale gold 386M genera COBOL, cobc lo compila
# ed esegue il caller ufficiale COBOLEval. Niente risposte scriptate.
#
# La sessione viene registrata in tempo reale dall'inizio (niente Hide/gate):
# si vede il caricamento pesi, poi i due task con `clear` tra l'uno e l'altro.

Output assets/skylar-cobol-demo.gif

Set Shell bash
Set FontSize 18
Set Width 1180
Set Height 880
Set Padding 24
Set Theme "Catppuccin Mocha"
Set TypingSpeed 20ms
Set PlaybackSpeed 1.0

Type "bash /home/mwspace/htdocs/skylar/pip-skylar/assets/run_cobol_demo.sh"
Sleep 600ms
Enter

# everything below is real-time: load banner -> task1 (clear) -> task2 (clear)
Sleep 22s
