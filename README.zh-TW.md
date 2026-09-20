# BreezeSprint25｜Breeze-ASR-25 × Colab TPU

> **v0.1.1 修正：**支援官方 `<|nocaptions|>` token，修復初始化時的 `Missing special token: <|nospeech|>`。
> 請同步更新原始碼與 Notebook；舊版 Notebook 內嵌的程式不會因 repo 更新而自動更換。[修復與重跑說明](docs/TOKENIZER_FIX.md)。

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thc1006/breezesprint-25/blob/main/BreezeSprint25.ipynb)

**臺灣華語與中英混用。兩格操作，上傳後自動下載時間戳 TXT。**

> **實驗性移植，尚未完成這個 Breeze 模型的真實 TPU 轉錄測試。**
> WhisperSprint 的成功紀錄不能直接當成 Breeze 的驗收。詳見 `evidence/validation.json`。

這是 ASR-25 華語／中英混用版本，不是 ASR-26 台語版本。偏繁體中文是模型特性，不保證每個字一定是繁體；不使用 OpenCC 或 LLM 改寫。

## 使用

第一格準備環境及模型；第二格只提供上傳，完成後每份錄音自動下載一份
`原檔名_transcript.txt`。不用挑模型、填路徑、掛載 Drive 或解壓縮。
下一份錄音重跑第二格即可。瀏覽器可能要求允許下載。

在 Colab 選擇 TPU v5e-1；免費額度與分配由 Google 決定。支援單裝置 v5/v6
的執行檢查，並不代表各型號均已實測。不要在同一個 runtime 同時啟動兩個版本。

## 不是只換模型名稱

兩款 Breeze 都是 Whisper-large-v2 架構：80 mel bins、32 層 encoder、32 層 decoder。
本版固定 `zh` + `transcribe`；這是應用程式的中文輸出選擇，不是假稱重現官方評測設定。
保留 448 token context，模型 token loop 留在 TPU，使用 BF16 與常駐快取。
不減速率、不刪重複、不用 LLM 潤稿。輸出本身仍然可能辨識錯誤。

25 使用 BF16 單檔，26 使用 F32 雙分片。程式會逐一檢查權重、形狀、分片索引、
固定 commit 與下載檔 ETag；用官方詞彙表解碼，不借用 Turbo 的 tokenizer。
模型下載約 3.09 GB，zip 內不附權重。

## 驗收界線

真實 TPU 運算與權重 schema 檢查通過後，第一次非零音訊視窗會跑兩次、比對 token。
**這只檢查執行及同設定重現性，不是對照人工正解，更不是中文／台語準確率證明。**
不再使用英文 JFK 測試來卡住台語專用模型。靜音檔可能不執行模型，報告不會冒充通過。

段落時間不是逐字對齊。26 的中文輸出也不能假稱逐字保留所有台語用詞。
任何報告、數字、專名或字幕正式發布前仍需人工聽校。

只有完成的 TXT 自動下載；JSON、進度與錯誤 log 留在 Colab 本機供除錯。
雲端運算不是完全離線，錄音會上傳 Google Colab。錯誤不會靜默回退 CPU 或其他 API。

原始碼、完整說明及測試指令見 [英文 README](README.md)。發布見 [PUBLISH.md](PUBLISH.md)。
應用 MIT；模型 Apache-2.0。獨立整合，非 MediaTek／NYCU／Google／OpenAI 官方專案。
