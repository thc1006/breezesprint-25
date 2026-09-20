# Output contract

Only one timestamped UTF-8 `.txt` per completed input is automatically offered for
download. Each cue has a sequential integer, `HH:MM:SS,mmm --> HH:MM:SS,mmm`, then
text and a blank line. Repeated words, punctuation and numbers are retained as
recognized; no post-transcription rewriting. Example data are labeled examples.

Times are segment estimates. Fallback window times carry a local JSON flag and
must not be described as word alignment. Local JSON/JSONL and logs remain for
recovery, not automatic download. An error does not turn partial output into a
complete result. Browser policy can block downloads; the completed TXT remains
available in the Colab Files panel. Output can be wrong even when well formatted.
