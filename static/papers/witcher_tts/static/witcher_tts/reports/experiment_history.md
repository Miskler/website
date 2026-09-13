# Witcher TTS — история экспериментов

Связная статья по проекту: [witcher_tts_story.md](witcher_tts_story.md).

Последнее обновление: 2026-09-13

Этот файл — рабочая статья проекта. Каждая новая модель должна добавляться сюда с четырьмя обязательными полями: гипотеза, конфигурация, численный результат, прослушиваемый результат. Численный loss без WAV-оценки не считается доказательством качества.

## 1. Цель

Построить русскую озвучку персонажей The Witcher 3 по параллельным русским и английским репликам. На первом этапе модель должна:

- произносить заданный русский текст;
- сохранять голос выбранного персонажа;
- обобщать на новые русские фразы;
- позднее принимать английскую дорожку как источник просодии.

Исходные данные и исходный датасет не изменяются. Рабочий проект: `C:/witchervoicesdataset/witcher_tts`.

## 2. Датасет

Собран изолированный датасет на 25 083 записи и 10 персонажей:

`geralt`, `yennefer`, `regis`, `triss`, `cirilla`, `dijkstra`, `baron`, `zoltan_chivay`, `shani`, `vesemir`.

Основные артефакты:

- `manifests/manifest.jsonl`
- `manifests/train.jsonl`
- `manifests/validation.jsonl`
- `manifests/test.jsonl`
- `manifests/speakers.json`
- `manifests/prepare_report.json`

Диагностический набор: 20 реплик — 10 Geralt и 10 Yennefer. В нём `train`, `validation` и `test` намеренно совпадают: это тест способности модели запомнить малый набор, а не оценка обобщения.

Последние контрольные тексты:

- `Всё холоднее… Уже близко…` — входит в 20 реплик;
- `И почему я не удивлена? Геральт, Геральт, что ты надел?` — входит в 20 реплик;
- `Ты ведь понимаешь, что это плохая идея?` — отсутствует в 20 репликах и используется как unseen.

## 3. Исправление аудиопредставления

Первоначальный mel был рассчитан через `log1p(mel)`, тогда как Vocos ожидает натуральный логарифм magnitude-mel. Это давало монотонные писки/модемные артефакты.

Исправление:

```python
torch.log(mel.clamp_min(1e-7))
```

Кэш правильного представления: `cache/mel_v1`.

## 4. Baseline — сверточная модель

Архитектура: около 16.5M параметров.

- посимвольная embedding + Transformer text encoder;
- интерполяция текста на длину mel;
- Conv1d acoustic decoder;
- mel/duration/prosody/speaker heads;
- duration head не управляет реальным выравниванием.

Checkpoint полного обучения: `checkpoints/stage_a_vocos_latest.pt`.

Результат 100 000 шагов:

- validation loss: `2.5595`;
- raw mel L1: `2.5582`.

Прослушивание: голос персонажа узнаваем, но слова и слоги практически не сохраняются. Как финальная модель baseline отвергнут.

Важно: на диагностическом наборе из 20 реплик raw baseline при длительном overfit-тесте научился воспроизводить заученные фразы. Это доказывает, что малый набор можно запомнить; это не доказывает обобщение.

## 5. Raw Transformer Flow

Архитектура: `44 359 279` параметров.

- 6-layer text Transformer, dim 384;
- 8-layer acoustic Transformer, dim 512;
- cross-attention к тексту;
- speaker embedding;
- flow matching от noise к normalized log-mel;
- Euler sampling.

Checkpoint полного обучения: `checkpoints/transformer_full_latest.pt`.

Результат 100 000 шагов:

- validation flow loss: `0.3326`;
- train loss в финальном логе: около `0.4299`.

Прослушивание: результат лучше baseline. Есть речь, голоса Geralt и Yennefer различимы, но звучание не человеческое и содержание фраз не распознаётся. Raw Transformer оставлен как основной контрольный кандидат, но финальным не признан.

На 20 репликах raw baseline и raw Transformer, по пользовательской оценке, за примерно 3 000 шагов научились хорошо произносить свои заученные фразы. Это ключевой контрольный факт проекта.

## 6. Проверка количества flow-шагов

Один и тот же полный Transformer checkpoint проверен на тех же текстах с seed 42 и 12/32/64 Euler-шагами.

Результат:

- содержание речи практически одинаковое;
- 32 шага иногда дают более детальную акустику;
- 64 шага не возвращают слова.

Вывод: sampler не является главным узким местом. Увеличение числа flow-шагов не исправляет потерю текстового содержания.

## 7. Phoneme/alignment v2

Была добавлена готовая фонемизация `espeak-ng` через `phonemizer`, без самописного G2P.

Новые изолированные манифесты:

- `manifests/phoneme_v2`
- `manifests/phoneme_overfit_geralt_yennefer_20`

Вариант v2 включал:

- phoneme tokenizer;
- CTC acoustic alignment;
- duration regulator;
- `x0` reconstruction loss;
- flow loss.

Checkpoint: `checkpoints/transformer_aligned_overfit_20.pt`.

На 1500 шагах численные метрики выглядели хорошо:

- validation CTC: `0.026`;
- validation x0: `0.0047`;
- validation duration: `0.055`.

Но пользователь не смог связать услышанное ни с одной входной фразой. При этом первые две контрольные фразы действительно были частью 20-репличного train-набора.

### Почему низкий CTC оказался недостаточным

CTC в этой реализации смотрел на настоящий `target mel`, а не на сгенерированный mel. Поэтому alignment head мог хорошо читать исходное аудио, пока flow-generator продолжал выдавать бессмысленную речь. Это была teacher-forcing утечка и неправильный критерий успеха.

Вывод: v2 не доказал, что генератор использует фонемы. Более того, он не воспроизвёл уже доказанную способность raw-моделей запоминать 20 фраз.

## 8. Что уже доказано

1. Правильный log-mel устраняет исходный монотонный писк.
2. Вокодер Vocos работает с правильным представлением.
3. Speaker identity модель учит: голоса различимы.
4. Raw baseline и raw Transformer могут overfit-ить 20 реплик и запоминать фразы.
5. На полном датасете raw модели не обеспечили читаемое содержание.
6. 12/32/64 flow-шагов не являются причиной потери текста.
7. Низкий CTC на reference mel не означает, что генерация произносит текст.
8. Текущий phoneme/alignment v2 нельзя считать успешным улучшением.

## 9. Следующий эксперимент: контролируемая абляция

Не запускать полный датасет и не подключать английскую просодию, пока не пройдён контрольный overfit.

Порядок:

### A. Raw Transformer + только фонемизация

Сохранить raw acoustic path, заменить только grapheme tokenizer на `espeak-ng` phonemes. Обучить на тех же 20 репликах около 3 000 шагов и проверить две известные фразы.

Это отвечает на вопрос: ломает ли содержание сама фонемизация или проблема была в alignment/duration implementation.

### B. Добавить только `x0` loss

Повторить тот же тест, не меняя способ подачи текстовых признаков.

### C. Добавить alignment/duration

Сначала проверить teacher-forced alignment отдельно, затем predicted durations на инференсе. Нельзя считать alignment рабочим только по CTC-loss — нужен CTC/ASR-контроль на сгенерированном аудио или человеческое распознавание фразы.

### D. Полный датасет

Только вариант, который повторяет известные 20 фраз и показывает читаемую unseen-фразу, допускается к полному обучению.

## 10. Английская просодия

Английская дорожка будет подключаться после успешного text-to-speech baseline как отдельный prosody/reference encoder. Входом останутся:

```text
русский текст + ID русского персонажа + английская озвучка
```

Английский WAV должен управлять F0, энергией, паузами и ритмом, но не заменять русский content encoder. Сопоставлять английские и русские кадры напрямую нельзя из-за разной длины перевода.

## 11. Правило ведения истории

После каждого запуска добавлять запись:

```text
Дата:
Гипотеза:
Модель/config:
Данные и число шагов:
Checkpoint:
Validation metrics:
Какие WAV прослушаны:
Что пользователь услышал:
Решение: принять / отвергнуть / повторить:
Следующий эксперимент:
```

## 12. Experiment A — raw Transformer + фонемизация

Дата: 2026-09-08

Гипотеза: если оставить доказавший overfit raw Transformer без alignment/v2-изменений и заменить только grapheme tokenizer на готовые русские фонемы `espeak-ng`, способность запоминать 20 фраз сохранится. Это отделяет проблему фонемизации от проблемы самодельного alignment/duration пути.

Конфигурация: `configs/transformer_phoneme_raw_overfit_20.yaml`.

Данные: `manifests/phoneme_overfit_geralt_yennefer_20`, 20 реплик, 3 000 шагов, seed 42, flow sampling 32 шага.

Checkpoint: `checkpoints/transformer_phoneme_raw_overfit_20.pt`.

Финальная validation:

- mel L1: `0.1883`;
- duration: `0.00042`;
- speaker: `0.00075`.

Контрольные WAV:

- `samples/raw_phoneme_3000_32_geralt_dataset.wav`;
- `samples/raw_phoneme_3000_32_yennefer_dataset.wav`;
- `samples/raw_phoneme_3000_32_geralt_unseen.wav`;
- `samples/raw_phoneme_3000_32_yennefer_unseen.wav`.

Численный результат хороший. Качественная оценка этих WAV должна быть добавлена после прослушивания; только после неё принимается решение о следующем компоненте.

Качественная оценка пользователя: unseen-вариант оказался крайне хорошо различимым; Geralt unseen — почти идеальный по разборчивости с учётом артефактов. Варианты с двумя заранее известными dataset-фразами заметно хуже разбираются, хотя одна из них входит в 20 реплик.

Промежуточный вывод на тот момент был признан недействительным после обнаружения ошибки инференса; raw+phoneme ещё не получил корректной качественной оценки.

Следующая проверка: прогнать все 20 диагностических текстов, а не только две выбранные фразы, с одинаковым seed и зафиксировать, какие именно реплики читаются. После этого — короткий full-dataset raw+phoneme прогон; английскую просодию пока не подключать.

### Исправление оценки Experiment A

Обнаружен критический баг в `scripts/infer.py`: при phoneme tokenizer исходный русский текст не преобразовывался обратно через `espeak-ng`; слова напрямую передавались в phoneme vocabulary и становились `UNK`. Поэтому WAV `samples/raw_phoneme_3000_32_*.wav` и запущенная до остановки сетка `samples/raw_phoneme_3000_32_overfit20_grid/` непригодны для оценки содержания.

Само обучение на phoneme manifests было корректным. Инференс исправлен, проверка даёт `UNK=0` для контрольных фраз.

Корректные WAV после исправления:

- `samples/raw_phoneme_3000_fixed_32_geralt_dataset.wav`;
- `samples/raw_phoneme_3000_fixed_32_yennefer_dataset.wav`;
- `samples/raw_phoneme_3000_fixed_32_geralt_unseen.wav`;
- `samples/raw_phoneme_3000_fixed_32_yennefer_unseen.wav`.

Предыдущие пользовательские оценки phoneme-вариантов, включая «unseen почти идеальный», относятся к некорректному инференсу и должны быть исключены из выводов. Это касается также `samples/aligned_overfit_20_32_*.wav`; оценки raw grapheme baseline/Transformer, где этот баг отсутствовал, остаются действительными.

### Корректная пользовательская оценка после исправления

Пользователь услышал следующие соответствия:

- Geralt dataset input `Всё холоднее… Уже близко…`: слышится начало «всё холоднее», качество плохое, голос точно Geralt;
- Yennefer dataset input `И почему я не удивлена? Геральт, Геральт, что ты надел?`: слышится другая заученная фраза вроде «что угодно, лишь бы горячее. и много», голос точно Yennefer;
- Geralt unseen input `Ты ведь понимаешь, что это плохая идея?`: слышится известная Geralt-фраза «всё холоднее, уже близко», голос точно Geralt;
- Yennefer unseen input: голос похож на Yennefer, но содержание неразборчиво.

Вывод: raw+phoneme надёжно кодирует speaker identity, но текстовое условие часто заменяется speaker-specific utterance prototype из train-набора. Это text-conditioning collapse, а не проблема Vocos или числа flow-шагов. На полном датасете этот вариант пока не запускать.

Следующий кандидат: готовый multi-speaker VITS с G2P и MAS как независимый контроль text adherence на тех же 20 репликах. Английскую просодию подключать только после того, как этот контроль начнёт следовать русскому тексту.

### Уточнение протокола оценки 20-репличного overfit

Предыдущая проверка с произвольным unseen-текстом была слишком строгой для задачи, где модель видела только 20 обучающих реплик. Её нельзя использовать как тест способности к обобщению текста.

Для корректного контроля VITS и следующих моделей фиксируем раздельные тесты:

- 2–3 известных текста Геральта с speaker=Geralt;
- 2–3 известных текста Йеннифэр с speaker=Yennefer;
- те же тексты с переставленным speaker ID, чтобы проверить, что текст и голос управляются раздельно.

Новый контрольный прогон VITS на 20 репликах оценивается по этому протоколу. Unseen-текст оставляем отдельным exploratory-тестом и не используем как критерий overfit.

То же исправление обязательно применяется к ранее обученным `stage_a_vocos_latest.pt`, `transformer_full_latest.pt` и `transformer_phoneme_raw_overfit_20.pt`: для каждого кандидата повторить 3 текста на каждого из двух speakers в режимах matching и swapped speaker. Старые qualitative-выводы о text adherence помечены как требующие повторной проверки.

## 13. Experiment B — официальный multi-speaker VITS, overfit 20

Дата: 2026-09-08

Гипотеза: готовый VITS с MAS и multi-speaker conditioning должен дать более честный контроль text adherence, чем самодельные mel/flow-варианты. Русский текст подаётся как готовые IPA-символы `espeak-ng`, английский WAV на этом этапе не используется.

Репозиторий: официальный [VITS](https://github.com/jaywalnut310/vits), изолированный checkout `C:/witchervoicesdataset/third_party/vits`.

Конфигурация: `third_party/vits/configs/witcher_overfit_20.json`; 24 кГц, 80 mel bins, 2 speakers, batch 2, segment 8192, генератор около 39.7M параметров. Из-за несовместимости старого кода с Windows/Torch использован FP32 single-GPU режим; исходные данные не изменялись.

Данные и шаги: те же 20 реплик Geralt/Yennefer, `global_step=3000`. Из-за bucket sampler в эпохе было 12 шагов, поэтому остановка выполнена вручную сразу после сохранения целевого checkpoint.

Checkpoint: `third_party/vits/logs/witcher_overfit_20/G_3000.pth` и `D_3000.pth`.

Ориентиры loss: на step 0 mel `81.7366`, KL `151.0308`; на step 3000 mel `24.5297`, KL `0.4987`. Это подтверждает, что модель учится, но не является доказательством разборчивой речи.

Скорость: около 29.5 минут на 3000 шагов, примерно 0.59 секунды/шаг. VRAM держалась примерно на 2.8–3.3 ГБ из 8 ГБ; ограничение — старый последовательный FP32 train-loop, а не память.

Корректная матрица VITS: `samples/vits_overfit_20_g3000/` — 12 WAV и `metadata.json`: три известных текста каждого исходного speaker в matching и swapped режимах.

Та же матрица для прошлых моделей: `samples/known_text_grid/baseline/`, `transformer/`, `raw_phoneme/`; карта текстов и speaker ID лежит в `metadata.json` каждой папки.

### Пользовательская прослушка Experiment B и повторной матрицы

Пользователь прослушал все четыре группы:

- `known_text_grid/baseline`: голоса различимы, но речь и текстовая привязка отсутствуют;
- `known_text_grid/raw_phoneme`: matching source-speaker + source-text в большинстве случаев даёт узнаваемую речь, но при swapped speaker получается каша без устойчивой речи;
- `known_text_grid/transformer`: не слышно ни различий актёров, ни текста; результат похож на инопланетный музыкальный инструмент;
- `vits_overfit_20_g3000`: актёры различимы, но речи нет; основной звук похож на артефакт воспроизведения после OOM.

Текущий ranking по корректному overfit-протоколу: raw+phoneme — единственный кандидат с частично рабочим matching content, но speaker transfer не отделён от текста; baseline, full Transformer и VITS на 3k не достигли разборчивой речи. Английскую модальность к этим моделям пока не подключать.

### Исправление: первая старая матрица использовала не те checkpoints

Обнаружено, что `samples/known_text_grid/baseline/` и `samples/known_text_grid/transformer/` были сгенерированы из полнодатасетных checkpoints `stage_a_vocos_latest.pt` и `transformer_full_latest.pt`, а не из 20-репличных overfit checkpoints. Поэтому их отсутствие речи невалидно как проверка маленького эксперимента.

Корректно пересозданные матрицы:

- `samples/known_text_grid/baseline_overfit20/` из `checkpoints/overfit_geralt_yennefer_20.pt`;
- `samples/known_text_grid/transformer_overfit20/` из `checkpoints/transformer_overfit_geralt_yennefer_20.pt`.

Предыдущая оценка full checkpoints оставлена на диске, но помечена как диагностически несопоставимая с overfit-экспериментом.

### Финальная пользовательская оценка корректных overfit-матриц

После прослушивания корректных `baseline_overfit20` и `transformer_overfit20` пользователь подтвердил: разборчивой речи нет. Модели самостоятельно выучили узнаваемые черты голосов — хрип, шипение и общий тембр — но не устойчивое содержание; голос удаётся узнать только с учётом контекста.

Это объединяет все текущие варианты: speaker identity частично обучается, text-to-acoustic alignment/content не формируется. Сравнивать кандидатов по «качеству речи» на этих точках пока бессмысленно. Предыдущие единичные впечатления о слышимых фразах считаем недостаточно надёжными без корректной matching/swapped-матрицы.

Следующий шаг: отдельная диагностика content path и waveform decoder на одной тренировочной реплике, чтобы разделить проблему текстового условия, alignment/duration и вокодера. Новую большую архитектуру и английскую модальность до этой диагностики не добавлять.

### Уточнение по четырём `*_exact.wav`

Пользователь отдельно отметил четыре старых preview, в которых, несмотря на артефакты, слышна речь:

- `samples/overfit_geralt_exact.wav`;
- `samples/overfit_yennefer_exact.wav`;
- `samples/transformer_overfit_geralt_exact.wav`;
- `samples/transformer_overfit_yennefer_exact.wav`.

Проверка происхождения показала:

- baseline-файлы совпадают побайтно с matching-кейсами `samples/known_text_grid/baseline_overfit20/geralt_1_geralt.wav` и `yennefer_1_yennefer.wav`. Это не новый или более сильный baseline, а тот же результат для двух конкретных известных реплик;
- transformer-файлы имеют те же sample rate, длительности и mel-форму, что и matching-кейсы, а их mel корректно декодируется тем же Vocos. Они были записаны сразу после checkpoint `checkpoints/transformer_overfit_geralt_yennefer_20.pt` (step 1500), но не совпадают побайтно с поздней матрицей `transformer_overfit20`, потому что старый запуск использовал другой случайный flow-sampling state/режим инференса;
- перебор текущих `flow_steps` 12/32/64 и нескольких seed не воспроизвёл старые transformer WAV. Поэтому старые transformer exact нельзя приписывать качеству одной фиксированной точки матрицы, но и выбрасывать их как «невалидные» нельзя: это реальные outputs того же 20-replica checkpoint, которые пользователь слышит лучше остальных;
- все четыре файла остаются сохранёнными как лучшие qualitative-примеры текущей серии. Их корректная интерпретация: модель уже способна иногда выдать речь-подобный, speaker-specific output на конкретной memorized реплике, но это ещё не доказательство устойчивого text-to-speech и не обобщение на новые фразы.

Практический ranking после уточнения: `transformer_*_exact.wav` — лучшие субъективные previews для дальнейшей диагностики; baseline exact — полезный контроль, но не отдельный кандидат. Следующий тест должен повторить именно этот known-text matching-кейс несколькими seed и затем проверить остальные 20 реплик, не смешивая результаты с прежней матрицей.

### Контрольный seed-sweep Transformer exact

Для проверки стабильности запущен отдельный прогон без изменения checkpoint и исходных данных:

- checkpoint: `checkpoints/transformer_overfit_geralt_yennefer_20.pt`, step 1500;
- known matching texts: `0x00026947` (Geralt) и `0x000286fa` (Yennefer);
- seeds: `0, 1, 7, 42`;
- flow steps: `12` и `32`;
- устройство: CUDA;
- результат: `samples/diagnostic_transformer_seed_sweep_exact_v1/metadata.json` и 16 WAV.

Ни один из 16 новых WAV не совпал побайтно со старыми `transformer_*_exact.wav`, что подтверждает: старые файлы были сгенерированы с незаписанным случайным состоянием/режимом sampling. Теперь нужно прослушать эту маленькую контрольную сетку и отметить для каждой фразы: разборчивость, наличие словесного ритма, speaker identity и устойчивость между seed. Только после этого имеет смысл расширять прогон на остальные известные реплики.

### Проверка влияния длины preview

После прослушивания seed-sweep пользователь сообщил, что речь слышна у Yennefer, а Geralt в основном хрипит. До вывода о speaker-specific failure обнаружено важное различие: старый preview Geralt имел длительность `1.87` с при исходной русской реплике `3.05` с, тогда как Yennefer — `4.12` с при исходной реплике `6.79` с. Текущий `infer.py` при отсутствии reference audio задаёт длину эвристикой `75 ms × число символов`, а не duration head модели.

В `scripts/infer.py` добавлен отдельный `--duration-audio`: он использует WAV только для длины, без извлечения просодии. Пересозданы контрольные варианты для обеих matching-фраз на seed 42 и flow steps 12/32 в `samples/diagnostic_transformer_duration_v1/`. Geralt теперь длится `3.04` с, Yennefer — `6.77` с. Эти файлы нужно прослушать до решения, что проблема именно в голосе Geralt.

### Первый production-like тест без reference audio

Проверка показала, что duration head у raw Transformer не использовался в `infer.py`: длина задавалась только эвристикой по числу символов. Добавлен флаг `--auto-duration`, который суммирует предсказанные per-token durations и сам задаёт число mel frames.

На том же checkpoint, seed 42 и flow steps 32 модель предсказала:

- Geralt: `287` frames = `3.061` с;
- Yennefer: `662` frames = `7.061` с.

Это уже inference без reference audio и без `duration-audio`; WAV лежат в `samples/diagnostic_transformer_auto_duration_v1/`. Следующая пользовательская проверка — сравнить эти два файла с duration-controlled вариантами. Если разборчивость сохраняется, базовый content path и duration path работают, а дальше можно расширять matching/swapped-матрицу и переходить к более долгому обучению.

### Swapped speaker-контроль без reference audio

По запросу выполнена парная проверка с теми же текстами, checkpoint, seed `42`, flow steps `32` и `--auto-duration`; изменён только speaker ID:

- текст Geralt с голосом Yennefer;
- текст Yennefer с голосом Geralt.

Результаты: `samples/diagnostic_transformer_auto_duration_swapped_v1/`. Это проверяет, разделяются ли content и speaker conditioning после устранения ошибки длительности.

### Пользовательская оценка swapped speaker

В обоих swapped-вариантах пользователь услышал одновременное присутствие двух персонажей: голос исходной реплики доминирует, а выбранный speaker ID добавляет второй тембр.

Это подтверждает speaker leakage в 20-replica overfit: text encoder запомнил не только содержание, но и voice prototype конкретной фразы, а слабое speaker conditioning не может полностью его переопределить. При matching это не проявляется, потому что текст и speaker ID согласованы.

Следствие: 20-репличный эксперимент доказал работу content path и waveform decoder, но не disentanglement speaker/text. Следующий кандидат для настоящей проверки — full-dataset Transformer с auto-duration; после него swapped-контроль нужно повторить на нескольких репликах. Если leakage сохранится на полном датасете, усиливаем/перестраиваем speaker conditioning и loss, а не вокодер.

### Full Transformer: повторная проверка после исправления длины

Подтверждено, что `checkpoints/transformer_full_latest.pt` — уже обученный full-dataset Transformer: 10 speakers, `step=100000`. Ранее он оценивался через старую эвристику длины и потому не давал честного сравнения.

Сгенерированы четыре новых файла с этим checkpoint, `auto-duration`, seed `42` и flow steps `32`:

- `samples/diagnostic_transformer_full_auto_duration_v1/geralt_text_geralt_voice.wav`;
- `samples/diagnostic_transformer_full_auto_duration_v1/geralt_text_yennefer_voice.wav`;
- `samples/diagnostic_transformer_full_auto_duration_v1/yennefer_text_yennefer_voice.wav`;
- `samples/diagnostic_transformer_full_auto_duration_v1/yennefer_text_geralt_voice.wav`.

Это первая честная matching/swapped-проверка уже full-модели после исправления инференса; пользовательская оценка этих четырёх WAV ещё впереди.

### Пользовательская оценка full Transformer после исправления длины

Пользователь отметил: текст и speaker отделены, но во всех четырёх full-модельных примерах речь звучит как «китайская» — голосовой аппарат/ритм присутствуют, русские слова и слоги не распознаются.

Это разделяет проблему на две части:

- speaker conditioning работает;
- duration path после `auto-duration` работает приемлемо;
- text-to-frame/content alignment для русских токенов не работает.

Следовательно, дальнейшее увеличение числа шагов raw-grapheme Transformer не является приоритетным. Нужен отдельный alignment-контроль с русскими фонемами и явным monotonic text-to-frame path (готовый VITS/MAS-путь или эквивалентная alignment-модель). Английскую модальность и просодию пока не добавлять: текущий дефект возникает раньше, на связывании русского текста с акустическими кадрами.

### Experiment C — продолжение официального VITS/MAS до 4000 шагов

Использован готовый официальный VITS с phonemizer/G2P и встроенным MAS; исходные 20-replica checkpoint не перезаписывались. `G_3000.pth` и `D_3000.pth` скопированы в отдельный каталог `third_party/vits/logs/witcher_overfit_20_continuation_4000/`, после чего выполнено продолжение до контрольной точки `G_4000.pth` / `D_4000.pth`.

Параметры продолжения: batch 2, FP32, CUDA, те же 20 реплик и тот же phoneme filelist; сохранение только на 4000, чтобы не расходовать оставшееся место. После сохранения процесс остановлен. Полная matching/swapped-матрица с `G_4000.pth` сгенерирована в `samples/vits_overfit_20_g4000/` (12 WAV + `metadata.json`).

Это следующий честный alignment-контроль: сначала сравнить `vits_overfit_20_g4000` с предыдущим `vits_overfit_20_g3000`, затем решать, продолжать ли VITS до 10k+ шагов или менять только режим обучения/ресурсную утилизацию.

### Продолжение VITS после пользовательской оценки G4000

Пользователь услышал на `G_4000`: голос уже есть, отдельные русскоподобные звуки и слоги появляются, но в слова ещё не складываются. Это улучшение относительно `G_3000`, поэтому alignment-путь не признан провальным.

Запущено продолжение того же VITS/MAS от `G_4000` с целью контрольных точек `G_6000`, `G_8000`, `G_10000`; сохранение настроено раз в 2000 шагов, чтобы удержать расход диска под контролем. Текущий процесс продолжается в `third_party/vits/logs/witcher_overfit_20_continuation_4000/`; исходные `G_3000` и исходный каталог VITS не изменяются по смыслу эксперимента, новая конфигурация сохранена отдельно.

Пользователь справедливо отметил, что предыдущие overfit baseline/Transformer уже давали слушабельный matching-результат примерно к 3500 шагам, тогда как VITS на `G_4000` пока даёт только голос, отдельные русскоподобные звуки и слоги без слов.

Сравнение не полностью эквивалентно: предыдущие модели использовали внешне заданную длину и готовый Vocos, а VITS одновременно учит MAS-alignment, duration, acoustic prior/posterior и waveform generator. Тем не менее текущий результат VITS признаётся хуже по пользовательскому критерию. `G_6000` выбран ближайшей decision point; `G_10000` не считается гарантией успеха.

### VITS G6000 — контрольная точка

Продолжение дошло до `G_6000.pth` / `D_6000.pth`. После сохранения `G_6000` процесс был остановлен на шаге 6030, поэтому контрольная точка валидна и не прервана во время записи. Сгенерирована полная matching/swapped-матрица в `samples/vits_overfit_20_g6000/` — 12 WAV и `metadata.json`.

Пользовательская оценка G6000 завершена: результат всё ещё неприемлем — голос узнаваем, но разборчивой речи нет. Ветка официального VITS/MAS закрыта как отрицательный Stage-A эксперимент; новые чекпоинты и дальнейшее обучение этого запуска не планируются.

### Пользовательская оценка VITS G6000

На `G_6000` результат всё ещё неприемлемый: голос узнаваем, но разборчивой речи нет. Переход от `G_4000` к `G_6000` не дал нужного качественного скачка. Продолжение того же запуска до `G_8000`/`G_10000` отменено: дополнительные шаги в текущей постановке с 20 уникальными репликами, продублированными в train/val, с высокой вероятностью будут только усиливать запоминание и артефакты, а не исправят text-to-speech alignment.

### Очистка промежуточных артефактов

После завершения VITS-эксперимента удалены 70 воспроизводимых промежуточных checkpoint-файлов (около 32.6 ГБ): старые VITS-снимки до `G_3000` и smoke/benchmark-чекпоинты. Сохранены исходные `G_3000`/`D_3000`, контрольные `G_4000`/`D_4000` и `G_6000`/`D_6000`, основные Transformer/Stage-чекпоинты, датасет, feature cache, samples и отчёты. Это освобождает место без потери артефактов, необходимых для воспроизведения и написания статьи.

### Аудит VITS-пайплайна после G6000

Проверены все 20 строк overfit-набора: manifest и VITS filelist совпадают по количеству, speaker ID, phoneme-тексту и токенизации; все 20 аудиофайлов существуют, монофонические и имеют частоту 24 кГц. Batch- и single-phonemizer дают одинаковый результат после нормализации, поэтому обнаруженное предупреждение phonemizer о количестве слов не приводит к расхождению входных последовательностей.

На контрольной подвыборке из четырёх реплик для `G_3000`, `G_4000` и `G_6000` MAS-attention монотонно проходит от начала текста к концу аудио. Явного рассинхрона filelist/text/audio или полного отказа MAS не найдено; текущая проблема относится к качеству акустического/вокодерного обучения либо режиму генерации.

В `scripts/infer_vits_overfit.py` добавлены параметры `--noise-scale`, `--noise-scale-w` и `--length-scale`. Для отделения случайности от качества checkpoint сгенерирована детерминированная матрица `samples/vits_overfit_20_g6000_deterministic/` с `noise_scale=0` и `noise_scale_w=0`; пользовательская оценка этой матрицы ещё впереди.

### Уточнение статуса full/phoneme Transformer

Полный 10-speaker Transformer действительно обучался до 100 000 шагов (`checkpoints/transformer_full_latest.pt`), но это был raw/grapheme-вариант: его конфигурация не содержит `text.tokenizer: phoneme`. Фонемный raw-контроль (`transformer_phoneme_raw_overfit_20.pt`) и aligned v2 (`transformer_aligned_overfit_20.pt`) обучались только на 20 репликах. Поэтому комбинация `полный 10-speaker dataset + phoneme input + raw acoustic path` в проекте ещё не запускалась; именно её ранее было ошибочно описано как уже выполненную.

### Уточнение уровня проблемы

До отдельной оценки русского произношения дело ещё не дошло. Текущие модели в основном научились воспроизводить тембр, шумовые/гармонические признаки и иногда речеподобный ритм, но не показали устойчивого text-to-acoustic content path. Поэтому корректная формулировка текущей проблемы — отсутствие стабильной речи и текстово-акустического выравнивания, а не ошибка произношения отдельных русских фонем. Phoneme-вход является следующим диагностическим упрощением задачи, но не считается уже найденным решением.

### Custom Transformer — teacher-forced акустический аудит

Для `checkpoints/transformer_full_latest.pt` проверены два настоящих target-mel: Geralt (`0x00026947`, 286 frames) и Yennefer (`0x000286fa`, 637 frames). Target-mel успешно декодируется frozen Vocos; контрольные WAV сохранены в `samples/diagnostic_custom_transformer_target/`.

При подаче настоящего target-mel в full raw/grapheme Transformer средний flow-loss относительно нулевого предсказателя составил `0.321x` для Geralt и `0.166x` для Yennefer. Предсказание общей длины также близкое: `0.991x` и `1.020x` от target frames. Это показывает, что Vocos, mel-конвенция и акустическая flow-цель работоспособны, а duration head обучился. Основной провал остаётся в свободном sampling из шума и/или в text-to-mel content path; до дальнейшего обучения нужен sampling-контроль на фиксированной истинной длине.

### Custom Transformer — sampling при истинной длине

Свободная генерация повторена для тех же двух target-текстов с точно заданным количеством target frames, чтобы полностью убрать duration head из эксперимента. Для Geralt normalized mel L1 составил `0.681`/`0.679`/`0.701` на 12/32/64 flow steps; для Yennefer — `1.319`/`1.374`/`1.412`. Увеличение числа sampling steps проблему не исправило. Следовательно, текущий full raw/grapheme Transformer не умеет восстановить содержательный mel из текста даже при правильной длине; проблема находится в text-to-mel content path/обобщении, а не в duration head или Vocos.

### Experiment D — продолжение full Transformer с 100k до 200k

Подтверждено, что `transformer_full_latest.pt` имеет `step=100000`, `vocab_size=145` и является raw/grapheme-моделью: в её конфигурации отсутствует phoneme tokenizer. Контрольная копия сохранена отдельно как `checkpoints/transformer_full_100k_grapheme_raw_frozen.pt`; SHA-256 исходника и копии совпадает (`D7C9BF105326C96842D7A04AA1A9F153130A94C8B567B03343358131B9119CA3`).

Для продолжения создан явный конфиг `configs/transformer_full_grapheme_continue_200k.yaml`: тот же 10-speaker dataset, та же архитектура Transformer Flow, `tokenizer: grapheme`, цель `step=200000`. Чтобы не было скачка learning rate после завершившегося 100k cosine schedule, продолжение использует `learning_rate=2e-5`; источник — только frozen-копия 100k, результат пишется в отдельный `checkpoints/transformer_full_200k_grapheme_raw_latest.pt`.

Первая попытка запуска остановилась на первом backward с CUDA `Memory allocation failure`; модель и источник не изменены, целевой checkpoint не создан. Повторный запуск стартовал успешно с отдельными файлами `logs/transformer_full_continue_100k_200k_retry1.csv` и `logs/transformer_full_continue_100k_200k_retry1.log`. SQLite в этом эксперименте не используется: существующие CSV/JSON-логи считаются достаточными.

### График обучения full Transformer 0–200k

Для контроля продолжения объединены только два чистых источника: `logs/overnight_transformer.log` для шагов 0–100k и `logs/transformer_full_continue_100k_200k_retry1.csv` для продолжения. Смешанный `stage_a_v2.csv` намеренно исключён. На графике синим показана исходная 0–100k-фаза, оранжевым — продолжение после 100k; текущий срез построен до шага `116000`. Файл: `reports/plots/transformer_full_0_200k_training_phased.webp`, численная сводка: `reports/plots/transformer_full_0_200k_training_phased_summary.json`.

### Text-conditioning ablation на frozen Transformer 100k

Продолжение обучения остановлено по запросу пользователя. Последняя безопасная сохранённая контрольная точка отдельной ветки — `checkpoints/transformer_full_200k_grapheme_raw_latest.pt` с `step=123000`; frozen-копия 100k не изменялась. Для абляции использована именно `checkpoints/transformer_full_100k_grapheme_raw_frozen.pt` с SHA-256 `D7C9BF105326C96842D7A04AA1A9F153130A94C8B567B03343358131B9119CA3`.

На 64 стратифицированных validation samples для каждого sample фиксировались `target_mel`, `noise`, `t` и `speaker_id`; менялся только текст: правильный, shuffled и blank/PAD. Зоны траектории: `t=0.15` (target-dominant), `t=0.50` (middle), `t=0.85` (noise-dominant; в реализации `t=0` — target, `t=1` — noise).

Результат flow-L1 и text dependency `(L_variant - L_correct) / L_correct`:

- target-dominant: shuffled `+0.18%`, blank `+4.18%`;
- middle: shuffled `+1.96%`, blank `+10.28%`;
- noise-dominant: shuffled `+11.69%`, blank `+16.60%`.

Вывод: текстовая conditioning-ветка не мёртвая — при отсутствии акустической подсказки сеть действительно ухудшает prediction. Но зависимость от конкретного русского текста слабая, особенно по сравнению с ожидаемым сильным content signal; модель использует текст частично и в основном опирается на акустический вход/общую статистику. Это не подтверждает гипотезу, что простое продолжение текущего objective само по себе решит разборчивость.

Дополнительно выполнен controlled free-sampling test: Geralt, одинаковые seed `501234`, длина `531` frames, `12` flow steps; варианты correct/shuffled/blank. WAV и mel сохранены в `samples/text_conditioning_ablation_100k/`. Полный отчёт: `reports/text_conditioning_ablation_100k/text_conditioning_ablation_100k.json`, таблица: `reports/text_conditioning_ablation_100k/text_conditioning_ablation_100k.csv`.

### Text-conditioning ablation на checkpoint 123k

Тот же эксперимент повторён без переобучения на сохранённом `checkpoints/transformer_full_200k_grapheme_raw_latest.pt` (`step=123000`): те же 64 stratified validation samples, seed `1234`, batch size `4`, зоны `t=0.15/0.50/0.85`, одинаковые `target_mel/noise/t/speaker_id` внутри каждого сравнения, варианты correct/shuffled/blank.

Text dependency `(L_variant - L_correct) / L_correct`:

- target-dominant: shuffled `+0.21%`, blank `+5.46%`;
- middle: shuffled `+2.49%`, blank `+12.52%`;
- noise-dominant: shuffled `+14.19%`, blank `+20.87%`.

Относительно 100k conditioning усилился, но остаётся умеренным: специфический чужой текст даёт лишь `+14.19%` flow-loss в noise-dominant зоне, а blank — `+20.87%`. Это подтверждает, что дополнительные 23k шагов немного усилили использование текста, но не превратили его в сильный content/alignment-сигнал.

Controlled free-sampling повторён с теми же параметрами; mel-L1 к correct output: shuffled `1.104`, blank `1.288` против `0.990` и `1.212` на 100k. WAV сохранены в `samples/text_conditioning_ablation_123k/`. Отчёт: `reports/text_conditioning_ablation_123k/text_conditioning_ablation_123k.json`, таблица: `reports/text_conditioning_ablation_123k/text_conditioning_ablation_123k.csv`.

### Пользовательская слуховая оценка text ablation

Пользователь отметил, что `geralt_correct.wav` можно связать с поданным текстом хотя бы по структуре фразы: слышимый контур/ритм/длина и расположение элементов напоминают именно фразу «Да… И нам тоже надо заняться этим всерьёз. Дело зашло слишком далеко.». Это не является разборчивым произношением слов, но даёт более сильное качественное подтверждение text conditioning, чем один flow-L1: модель, вероятно, уже использует текст для структуры высказывания, одновременно сохраняя нерешённой проблему фонемного/словесного content alignment.

Уточнение пользователя: вариант с чужим текстом «Ни хера!» нельзя связать с услышанным аудио вообще. Таким образом, различие не сводится только к общей случайной вариативности: correct-вариант сохраняет хотя бы структурное соответствие исходной фразе, тогда как shuffled-вариант его теряет. Это усиливает вывод о реально работающем, но пока крупнозернистом text conditioning.

Дополнительное наблюдение: `blank` звучит как тихое хрипение и вздохи. Все три controlled free-sampling WAV имеют одинаковую длительность около 5.66 с намеренно: в тесте для изоляции text conditioning принудительно использовались одни и те же `531 frames`, seed, speaker и flow steps. Поэтому длина blank не является предсказанием duration head; это зафиксированная длина эксперимента.

### Experiment E — auxiliary text-ranking loss от 123k до 200k

Контрольная копия остановленной модели сохранена как `checkpoints/transformer_full_123k_grapheme_raw_frozen.pt`, `step=123000`, SHA-256 `343C3D0FED75C96EC571B903BA2879A1043EE42AD3971D2E9623BFDC5E432771`. Исходный `transformer_full_200k_grapheme_raw_latest.pt` также не перезаписывается.

Добавлен auxiliary loss для text conditioning. На одном и том же `x_t`, `noise`, `t` и `speaker_id` считаются flow-L1 для correct и same-batch shuffled текста. Для `t >= 0.60` оптимизируется мягкое ranking-условие, требующее `L_shuffled` быть выше `L_correct` минимум на относительный margin `0.10`:

`D_text = (L_shuffled - L_correct) / (L_correct + epsilon)`;

`L_text = softplus((0.10 - D_text) / 0.05)`.

Общий loss получает `0.10 * L_text`; исходные flow, duration и speaker-компоненты сохранены. Чтобы оставить запас VRAM для второго forward, frame budget снижен с `6000` до `3000`. В CSV добавлены `text_rank`, `text_dependency`, `text_correct`, `text_shuffled`, `text_samples`; SQLite не используется.

Одношаговый smoke-тест прошёл на CUDA. Основной отдельный запуск стартовал от frozen `123000` с целью `200000`; его артефакты: `checkpoints/transformer_full_textloss_200k_latest.pt`, `logs/transformer_full_textloss_123k_200k.csv`, `logs/transformer_full_textloss_123k_200k.log`. Первые активные batch-и показали `text_dependency` около `0.11–0.21`; batch-и без двух подходящих samples пропускают auxiliary loss без ошибки.

### Experiment E — режим повышенной утилизации

После проверки ресурсов ветка с `frame_budget=3000` остановлена на сохранённой точке `step=126000`; она сохранена как `checkpoints/transformer_full_textloss_126k_frozen.pt`. Потеряны только несохранённые шаги после последнего periodic checkpoint.

Создан отдельный оптимизированный конфиг `configs/transformer_full_textloss_optimized_from_126k.yaml`: `frame_budget=5000`, `num_workers=2`, `pin_memory=true`, `prefetch_factor=2`, тот же auxiliary text-ranking loss и тот же target `200000`. Новый запуск стартовал от frozen `126000` в `checkpoints/transformer_full_textloss_optimized_200k_latest.pt`; CSV и stdout: `logs/transformer_full_textloss_optimized_126k_200k.csv` и `logs/transformer_full_textloss_optimized_126k_200k.log`.

Первые замеры оптимизированной ветки: GPU utilization `71–86%`, power `68–75 W`, VRAM около `2.8 GiB`, OOM не наблюдается. Это выше предыдущих `67–81%` при `frame_budget=3000`, при этом остаётся запас памяти для стабильного продолжения.

При последующей проверке на более длинном batch тот же режим `frame_budget=5000` достиг `89% GPU utilization` и `7894/8151 MiB VRAM` на шаге около `126850`. Поэтому ранний замер `2.8 GiB` был только коротким малым batch; увеличивать frame budget дальше нельзя без высокого риска CUDA OOM. Текущая оптимизированная ветка оставлена работать без дополнительного перезапуска.

Уточнение по скорости: рост GPU utilization не означал ускорение по шагам. По временным меткам логов ветка `frame_budget=3000` прошла примерно 3000 шагов за 5:53 (`~8.5 step/s`), тогда как `frame_budget=5000` прошла около 1850 шагов за 6:49 (`~4.5 step/s`). Второй forward для text-ranking и больший batch сделали один шаг существенно дороже. Поэтому `5000` лучше утилизирует GPU, но может быть хуже по wall-clock ETA; для честного выбора нужно сравнивать mel-frames/sec и итоговое качество, а не только GPU utilization.

### Возобновление после закрытия приложения

Закрытие приложения Codex завершило дочерний процесс обучения: это был обычный background-процесс, а не системный Windows daemon. Последняя безопасная periodic checkpoint-точка оптимизированной ветки — `step=128000` в `checkpoints/transformer_full_textloss_optimized_200k_latest.pt`.

Перед перезапуском checkpoint зафиксирован отдельной frozen-копией `checkpoints/transformer_full_textloss_128k_frozen.pt`; SHA-256 исходника и копии совпадает: `BEF37D63B2E722769F918DE7A39133D699B3A462E5619D79D205BEAE33A04B69`. Исходный checkpoint, датасет и старый CSV не изменялись.

Обучение возрождено от `step=128000` до `step=200000` отдельным скрытым процессом. Новые результаты пишутся в `checkpoints/transformer_full_textloss_optimized_200k_resumed_latest.pt`, `logs/transformer_full_textloss_optimized_128k_200k_resumed.csv` и соответствующий `.log`; SQLite не используется. После запуска процесс жив, CUDA-контекст создан, первые строки нового CSV появились на шагах `128100–128200`.
### График auxiliary text loss на возрождённой ветке

Для возрождённого запуска от frozen `step=128000` построен текущий срез по активным batch-ам с `text_samples > 0`, от `128100` до `138250`. Основной график показывает `D_text = (L_shuffled - L_correct) / L_correct`, а второй — фактический auxiliary `text_rank`; добавлено 15-точечное сглаживание. PNG/SVG и сводка сохранены в `reports/plots/text_conditioning_loss_128k_200k_resumed.*`.

На этом срезе средний `D_text` составляет около `0.243`, последнее сглаживаемое значение около `0.346`; сигнал текстовой зависимости присутствует, но сильно шумит между batch-ами. Это диагностический график, а не evidence разборчивой речи: он показывает, использует ли objective правильный текст, но не измеряет фонемную точность.
### Диагноз: почему auxiliary text loss не превращается в речь

Проверка реализации показала, что проблема не в недостатке ещё нескольких десятков тысяч шагов. Текущая `transformer_flow`-ветка оптимизирует flow-L1 по mel и speaker classification, но не имеет явного text-to-frame alignment/CTC/phoneme-duration objective. В `TransformerFlowStageAModel.forward()` `target_frames` удаляется, а `sample()` получает длину извне; поэтому наблюдаемая одинаковая длительность около 5 секунд — следствие inference-настройки, а не выученной длительности реплики.

`duration_head` обучается на искусственной равномерной цели `target_frames / text_length`, но его выход не подаётся в акустический путь и не раскладывает текст по mel-фреймам. Существующий `transformer_flow_aligned` с CTC/forced alignment в текущем конфиге не используется: указано `architecture: transformer_flow`.

Добавленный `D_text` исправляет только относительное ранжирование: на одном `x_t` correct-text должен иметь меньший flow-loss, чем shuffled-text. Это не учит, какой фонеме соответствует какой участок аудио, и активируется только при `t >= 0.60`, обычно на 2–8 samples одного batch-а. Поэтому сигнал получается шумным и может улучшать различимость условий без появления слов. На свежем срезе `D_text` не растёт (`0.307` в первых 40 active batch-ах против `0.244` в последних), а `text_rank` вырос (`0.966` против `1.240`).

Вывод: текущая модель стала лучше учить акустический/спикерный proxy, но objective не предъявляет ей достаточно прямого требования произносить конкретные русские фонемы в конкретной временной последовательности. Продолжение до 200k полезно как контрольная ветка, но не является исправлением этой архитектурной ошибки.
### План следующего эксперимента: исправление text-to-speech alignment

Текущая `transformer_flow`-ветка продолжает обучение только как контрольный эксперимент до `200000` шагов. Её не считать финальным кандидатом: рост числа шагов не устраняет отсутствие text-to-frame alignment.

Цель следующей ветки — проверить именно произношение, а не ещё раз общий тембр:

1. Переключить архитектуру на уже существующий `transformer_flow_aligned` с CTC/forced alignment.
2. Переключить текстовый вход с grapheme на уже подготовленный phoneme tokenizer и использовать `ru_phoneme_vocab.json`.
3. Инициализировать совместимые веса из текущей Transformer-модели через `--init-checkpoint`; новые alignment-слои оставить с нуля. Не продолжать optimizer state текущей raw/grapheme-ветки.
4. Не добавлять на первом прогоне `D_text`: сначала проверить, что CTC/alignment действительно привязывает фонемы к mel-фреймам.
5. Провести короткий overfit-тест на небольшом фиксированном наборе Geralt/Yennefer: для каждого спикера несколько известных реплик, затем проверить те же тексты с правильным и чужим speaker ID.
6. Критерии успеха: известный текст должен быть различим на слух, длительность должна зависеть от текста, а чужой speaker должен менять тембр без разрушения фонемной последовательности.
7. Только после успешного маленького теста запускать full 10-speaker dataset и сравнивать с frozen raw/grapheme Transformer.

Что не делать: не увеличивать размер сети и не продолжать добавлять ranking-loss, пока не доказано, что alignment-ветка учит произношение. Главная ошибка текущей ветки — не недостаток capacity, а отсутствие прямого требования «фонема → соответствующий временной участок аудио».
### Experiment F — aligned Transformer + phonemes, overfit-контроль

Текущая raw/grapheme ветка зафиксирована на `step=142000` и остановлена после записи periodic checkpoint. Frozen-копия сохранена как `checkpoints/transformer_full_textloss_142k_grapheme_raw_frozen.pt`; SHA-256: `25D36A67D1E2DC0F89246CF6C36F679AA96B1FA4EE792D08A41EF9DDCADAD888`. Ранее сохранённые checkpoint и CSV не изменялись.

Запущен отдельный короткий тест уже существующей архитектуры `transformer_flow_aligned` с phoneme tokenizer и полным CTC/forced-alignment objective. Используются готовые `configs/transformer_aligned_v2.yaml` и `manifests/phoneme_overfit_geralt_yennefer_20`: 20 реплик, по 10 Geralt и Yennefer, phoneme vocabulary — 68 токенов. Совместимые веса из frozen raw-модели подгружены через `--init-checkpoint` (`285/318` tensors); alignment-слои и phoneme embedding инициализированы отдельно. `D_text` в этом тесте намеренно не используется.

Результат запуска: отдельный процесс CUDA жив, артефакты пишутся в `checkpoints/transformer_aligned_phoneme_overfit_20_latest.pt` и `logs/transformer_aligned_phoneme_overfit_20.csv`. На step `1 → 50` CTC-loss снизился примерно `42.17 → 3.61`; это ранний диагностический сигнал, не доказательство слышимой речи. После завершения нужно сгенерировать known-text grid и проверить разборчивость, длительность и перенос голоса между двумя персонажами.
### Experiment F — ускоренный режим aligned phoneme overfit

Первый aligned-прогон был зафиксирован на `step=500` (`checkpoints/transformer_aligned_phoneme_overfit_20_step500_frozen.pt`) и остановлен, чтобы не ждать полный тест в медленном режиме. Создан отдельный конфиг `configs/transformer_aligned_phoneme_overfit_20_fast.yaml`: `frame_budget=3000`, `num_workers=2`, `prefetch_factor=2`, `pin_memory=true`; `D_text` по-прежнему отключён.

Обучение продолжено с frozen `step=500` в отдельный checkpoint `checkpoints/transformer_aligned_phoneme_overfit_20_fast_latest.pt` и отдельные fast-логи. Новый процесс CUDA жив; после переключения скорость выросла примерно с `0.77` до `1.31 step/s`, ожидаемое завершение 1500 шагов — около `22:06`. Снижение frame budget сделано только ради быстрого диагностического прогона, не как финальная настройка full dataset.
### Сравнение aligned phoneme Transformer с raw phoneme Transformer

Сгенерирован одинаковый known-text grid из 12 cases для двух моделей: первые три реплики Geralt и первые три Yennefer, каждая с обоими target speakers. Для обеих моделей использованы одинаковые phoneme texts, seeds, 12 flow steps и длина исходного RU-аудио.

Сравнены `transformer_flow_aligned` на `step=1500` и `transformer_flow` raw phoneme на `step=3000`. Аудио сохранено в `samples/compare_aligned_vs_raw_phoneme/aligned/` и `samples/compare_aligned_vs_raw_phoneme/raw_phoneme/`, сводка со ссылками — `reports/compare_aligned_vs_raw_phoneme.md`. Разница в количестве шагов отмечена: это архитектурное сравнение, но не строго равный budget обучения.
### Слуховой результат aligned vs raw phoneme

Пользователь отметил, что в текущем known-text grid aligned-версия говорит заметно хуже raw phoneme Transformer. Поэтому `transformer_flow_aligned` пока не считается улучшением и не запускается на full dataset без дополнительной проверки.

Сравнение было полезным, но не полностью равным: aligned обучался до `1500` шагов и стартовал из raw/grapheme checkpoint, тогда как raw phoneme имел `3000` шагов и полноценные phoneme-веса. Кроме того, в текущей aligned-реализации есть train/inference mismatch: при обучении frame-level text строится из CTC forced alignment, использующего target mel, а при инференсе — из предсказанных duration. Акустический путь видит идеальное выравнивание на train и ошибочное/необученное duration-выравнивание на inference.

Следующий честный тест: инициализировать aligned из `transformer_phoneme_raw_overfit_20.pt`, дать обеим моделям одинаковый budget до `3000` шагов и отдельно проверить fixed-duration и auto-duration. Full dataset до этого не запускать.
### Experiment G — aligned phoneme от raw phoneme до 4000 шагов

После того как aligned-модель на `1500` шагах проиграла raw phoneme baseline, запущен честный повтор. `transformer_flow_aligned` получает совместимые веса из `checkpoints/transformer_phoneme_raw_overfit_20.pt` через `--init-checkpoint`: совпадающие phoneme/text/acoustic tensors загружены (`286/318`), alignment-слои и остальные несовместимые части обучаются с новым optimizer state.

Цель — `4000` шагов на том же `phoneme_overfit_geralt_yennefer_20`, без `D_text`, с fast-настройками `frame_budget=3000`, двумя workers и pinned memory. Результаты пишутся отдельно в `checkpoints/transformer_aligned_phoneme_from_raw_4k_latest.pt` и `logs/transformer_aligned_phoneme_from_raw_4k.csv`; предыдущие aligned/raw checkpoints не изменяются. На первых 50 шагах CTC-loss снизился `38.87 → 3.97`.
### Experiment G — результат aligned phoneme от raw до 4000 шагов

Эксперимент завершён штатно на `step=4000`; процесс CUDA остановился после validation и записал `checkpoints/transformer_aligned_phoneme_from_raw_4k_latest.pt`. Финальная validation: общий loss `0.1101`, flow/mel L1 `0.1045`, duration `0.0413`, speaker `0.000089`, x0 `0.00265`, CTC `0.000600`. На последнем train batch CTC был `0.00327`.

Это означает, что на маленьком 20-репличном наборе alignment/CTC сошёлся численно. Следующая обязательная проверка — повторно сгенерировать тот же known-text grid для этого `4k` checkpoint и сравнить его с raw phoneme Transformer; низкий CTC-loss сам по себе ещё не доказывает слышимую разборчивость.
### Known-text grid для aligned checkpoint 4000

После завершения Experiment G заново сгенерирован тот же 12-case grid для `transformer_aligned_phoneme_from_raw_4k_latest.pt`. Условия оставлены идентичными предыдущему сравнению с raw phoneme: одинаковые phoneme texts, seed sequence, 12 flow steps и длительность исходных русских записей.

Новые WAV сохранены в `samples/compare_aligned_4k_vs_raw_phoneme/aligned_4k/`; сравнительный отчёт со ссылками — `reports/compare_aligned_4k_vs_raw_phoneme.md`. Raw phoneme reference взят из ранее сохранённого `samples/compare_aligned_vs_raw_phoneme/raw_phoneme/`.
### Пользовательская оценка aligned 4k и запуск full dataset

Пользователь отметил по aligned 4k grid: артефакты при воспроизведении исчезли, но cross-speaker condition не работает — реплика воспроизводится голосом исходного персонажа, а не выбранного target speaker. Это зафиксировано как главный критерий full-dataset обучения: сохранить фонемную разборчивость и одновременно добиться управления speaker ID.

Источник full-запуска заморожен как `checkpoints/transformer_aligned_phoneme_from_raw_4k_frozen.pt`, SHA-256 `D5BCEDDCAC668A133026A3D8652DF09ADC1CFA178F2D31C3241896A41DA6B54C`. Запущен отдельный full aligned+phoneme run на `manifests/phoneme_v2` (22570 train records, 10 speakers), без `D_text`, с целью `100000` шагов. После короткого benchmark frame budget `12000` был отброшен из-за CPU-side forced-alignment bottleneck; рабочая ветка использует `frame_budget=5000`, 4 workers и pinned memory. Её артефакты: `checkpoints/transformer_aligned_phoneme_full_100k_fast2_latest.pt` и `logs/transformer_aligned_phoneme_full_100k_fast2.csv`.

На первых 50 шагах full-ветка жива, но GPU пока утилизируется импульсно и невысоко: CTC/forced-alignment реализован с последовательными Python/torch операциями по time axis и является текущим throughput bottleneck. Это отмечено отдельно для последующей оптимизации; процесс не считается завершённым или оценённым по качеству.

### Исправление throughput: native torchaudio forced alignment

Полный запуск был остановлен на примерно `step=400`; periodic checkpoint этой ветки ещё не успел записаться, поэтому исходной безопасной точкой остался frozen `transformer_aligned_phoneme_from_raw_4k_frozen.pt`. Причина низкой утилизации подтверждена: старый `ctc_forced_alignment` делал Python-цикл по каждому mel-фрейму с зависимыми маленькими CUDA-операциями и синхронизацией `.cpu()`. Это не требовало 100% CPU — один host-поток мог держать GPU в ожидании при низкой общей загрузке процессора.

В `src/alignment.py` старый DP заменён на готовый `torchaudio.functional.forced_align` (`torchaudio 2.11.0+cu130`), который выполняет forced alignment нативно на CUDA. Добавлен корректный перевод возвращаемого token-id path в индексы фонем, включая повторяющиеся фонемы; старый алгоритм оставлен fallback-веткой. `py_compile`, CUDA smoke-тест и 3 реальных train-шага на full aligned pipeline прошли; CTC и остальные метрики считаются штатно.

Временный smoke-checkpoint после проверки удаляется, чтобы не расходовать место. Full training перезапускается отдельным checkpoint/log после патча; frozen source и исходные датасеты не изменяются.

Короткое сравнение budget после патча: `frame_budget=12000` даёт заметные bursts до `100% GPU`, но увеличивает время одного шага непропорционально из-за квадратичной attention-нагрузки на крупном batch; поэтому для full-run оставлен `5000`, который даёт лучший throughput по обработанным mel-фреймам. Это не означает постоянные 100% графика GPU: у модели остаются короткие forward/backward и native CTC kernels, между которыми GPU ждёт host orchestration.

После перезапуска native full-run идёт на `frame_budget=5000`, `num_workers=4`, `pin_memory=true`, от frozen 4k-init. На контрольной точке около `step=300` parent training process использовал около `450 CPU-секунд` за `~92 wall-секунды` (примерно `4.9` логических ядра в среднем), GPU показывал bursts до `79%`; это заметно лучше диагностированного старого режима с почти одним CPU-ядром. Активные артефакты новой ветки: `checkpoints/transformer_aligned_phoneme_full_100k_native_latest.pt` и `logs/transformer_aligned_phoneme_full_100k_native.csv`. Первый periodic checkpoint ожидается на `step=1000`.

### График native full-run на 53.75k

По текущему CSV построен сводный график `reports/plots/transformer_aligned_phoneme_full_native_training.png/.svg`: total loss, flow/mel L1, CTC, duration, x0 и speaker loss; серым показаны train batch-ы, синей линией — сглаженный train, оранжевыми точками — validation. На `step=53750` train total loss около `0.478`, train CTC около `0.433`; последняя записанная validation на `step=53000`: total `0.574`, flow/mel `0.325`, CTC `1.146`. График показывает устойчивое снижение основных objective-метрик без явного развала validation, но не доказывает ещё слышимую разборчивость.

### Остановка на 54k и сравнение с ordinary Transformer 128k

По просьбе пользователя aligned full-run остановлен через `Ctrl+C` после сохранённого checkpoint `step=54000`; несохранённые шаги после этой точки отброшены. Обычная grapheme-модель frozen на `step=128000` сохранена отдельно и не изменялась.

Сгенерирован одинаковый known-text grid: 3 реплики Geralt и 3 Yennefer, каждая с target speaker Geralt/Yennefer, всего 12 случаев на модель. Для обеих веток совпадают исходные тексты, target frames, seeds `123–134`, Vocos и `12` flow steps. Aligned использует phoneme tokenizer, ordinary — grapheme tokenizer; генератор теперь выбирает tokenizer по config. WAV и таблица сравнения сохранены в `samples/compare_aligned_54k_vs_raw_128k/` и `reports/compare_aligned_54k_vs_raw_128k.md`.

По просьбе пользователя в ту же папку добавлены шесть исходных русских WAV: `samples/compare_aligned_54k_vs_raw_128k/original/geralt_1_original.wav` … `geralt_3_original.wav` и `yennefer_1_original.wav` … `yennefer_3_original.wav`. Они соответствуют тем же source IDs и длительностям, что использованы в generated grid.

График обучения обновлён после продолжения до `121950`: `reports/plots/transformer_aligned_phoneme_full_native_54k_200k_training.webp`. Для корректного полного вида объединены логи первой ветки `step 1–54000` и continuation `step 54050–121950`; показаны total, flow/mel, CTC, duration, x0 и speaker loss с train/validation.

### Анализ кривых на ~122k

На `0–20k` идёт основное быстрое обучение; на `20–55k` метрики продолжают заметно улучшаться; после `~80–100k` validation уже близка к плато. Validation total снизился с `1.031` на `1k` до минимума `0.514` на `119k`; на последней точке около `122k` — `0.543`. Validation flow/mel L1 достигал `0.282` на `119k` и был около `0.309` на последней точке. CTC продолжает медленно улучшаться: `2.819` на `1k` → минимум `1.059` на `119k`, затем около `1.081` на `122k`.

Train/validation gap уже есть: на последней точке total примерно `0.421` против `0.543`, CTC `0.406` против `1.081`, duration `0.058` против `0.108`. Это умеренное переобучение alignment/CTC на train, но не катастрофическое: validation ещё не деградирует системно, а колеблется вокруг плато. Вывод: продолжение до `200k` может дать небольшой прирост, особенно в CTC и устойчивости, но ждать пропорционального улучшения качества речи от дополнительных `80k` шагов не стоит; после `~120k` это уже diminishing returns. Для слуховой оценки важнее сравнить checkpoint около `100–120k` и финальный `200k` на unseen-фразах.

### Тест нестандартного текста после остановки на 54k

Для проверки обобщения без reference audio сгенерированы две новые фразы: `Саша шла по шоссе.` и `Ехал Грека через реку, видит Грека — в реке рак.`. Каждая фраза озвучена Geralt и Yennefer у aligned checkpoint `step=54000` и ordinary grapheme checkpoint `step=128000`; использованы `12` flow steps и одинаковые seeds/эвристика длительности. Результаты: `samples/nonstandard_compare_aligned_54k_vs_raw_128k/`, сводка: `reports/nonstandard_text_compare_aligned_54k_vs_raw_128k.md`.

### Решение после теста нестандартного текста

Aligned-модель уже переносит текст на unseen-фразы, но качество сильно недообученное. Полный phoneme train-manifest содержит `22570` реплик и `22.45 h` русского аудио: Geralt `15.9 h`, Yennefer `1.21 h`, остальные персонажи от `22.1` до `59.3` минут. Предложенный корпус «все speakers >30 минут + весь Geralt» почти совпадает по объёму: исключаются только Shani (`28.1 min`) и Vesemir (`22.1 min`), то есть около `2.2%` аудио; весь Geralt уже присутствует в текущем корпусе.

Рекомендация: пока продолжать именно текущую aligned-модель на полном корпусе, не меняя dataset и objective. Validation CTC на `54k` ещё снижался (`2.82` на `1k` → `1.15`), а total loss вышел лишь на плато, что больше похоже на недообучение, чем на устойчивое переобучение. Сначала продолжить до `100k–150k`, затем повторить тесты `Саша шла по шоссе` и «Грека/рак». Если после этого качество упрётся в потолок, сделать отдельную Phase B на отфильтрованном корпусе с контролируемым speaker sampling; не дублировать Geralt вручную.

### Продолжение aligned-модели до 200k

Пользователь выбрал продолжение текущей aligned-модели с квотой `200000` шагов. Запущен отдельный resume-run от frozen/последнего checkpoint `step=54000`; исходный `transformer_aligned_phoneme_full_100k_native_latest.pt` не перезаписывается. Новая ветка пишет в `checkpoints/transformer_aligned_phoneme_full_200k_latest.pt` и `logs/transformer_aligned_phoneme_full_native_54k_200k.csv`. Optimizer state восстановлен, cosine scheduler корректно создан заново с горизонтом `200k`; первый новый train-лог появился на `step=54050`.

### Остановка continuation и тест на 122k

По просьбе пользователя continuation остановлен через `Ctrl+C` во время шага около `122650`. Последний корректно записанный checkpoint — `checkpoints/transformer_aligned_phoneme_full_200k_latest.pt` с метаданными `step=122000`; исходные checkpoint `54k` и ordinary `128k` не изменены. Процесс обучения после остановки отсутствует.

Из checkpoint `122000` повторён тест на тех же двух unseen-фразах (`Саша шла по шоссе.` и `Ехал Грека через реку, видит Грека — в реке рак.`), для Geralt и Yennefer, с теми же seeds и `12` flow steps. Новые файлы имеют префикс `aligned_122k_` и лежат рядом с результатами `aligned_54k`/`ordinary_128k`; сравнительная таблица обновлена в `reports/nonstandard_text_compare_aligned_54k_vs_raw_128k.md`.

### Переход к фазе reference-audio / prosody conditioning

Текущая тестовая размерность признана достаточной для проверки основной гипотезы: aligned phoneme Transformer уже умеет формировать речь и обобщаться на нестандартный русский текст. Продолжение после примерно `122k` дало в основном небольшую стабилизацию без качественного скачка, поэтому дальнейшее наращивание этой тестовой модели остановлено. Checkpoint `step=122000` сохраняется как baseline и не перезаписывается.

Следующая гипотеза: модель должна получать интонацию, ритм и просодическую структуру из отдельного reference-аудио, но содержание генерировать по русскому тексту, а тембр — по выбранному персонажу. Reference-аудио не должно заставлять модель произносить исходный английский или польский текст; его роль — подсказать, как должна звучать фраза по интонации и времени.

#### Планируемые три модели

Будут обучены три одинаковые по архитектуре и размерности кандидата, различающиеся только составом языков reference-аудио:

1. `reference_en` — reference-аудио только из английской озвучки.
2. `reference_pl` — reference-аудио только из польской озвучки.
3. `reference_en_pl_ru` — смешанный reference-корпус: английская, польская и родная русская озвучка.

Во всех трёх вариантах целевой выход остаётся русским аудио. На входе сохраняются русский phoneme-текст и `target_speaker`; добавляется отдельная ветка reference-аудио с признаком языка. Для смешанной модели русское reference-аудио является контрольным режимом: оно позволит проверить, не ломается ли просодический перенос на родном языке, а английский и польский — проверить межъязыковое обобщение.

#### Порядок реализации и проверки

1. Зафиксировать текущий `transformer_aligned_phoneme_full_200k_latest.pt` (`step=122000`) и не использовать его как единственный результат: он остаётся baseline для сравнения без reference-аудио.
2. Собрать парные manifests по общему ID реплики: `reference_audio + reference_lang + русский text/phonemes + target Russian wav + target_speaker`. Проверить, что в validation нет утечки одной и той же реплики между train и validation.
3. Добавить reference-аудио encoder с bottleneck/conditioning в aligned Transformer. Текстовая ветка должна продолжать отвечать за фонемное содержание, speaker embedding — за голос персонажа, reference-ветка — за просодию.
4. На одинаковом маленьком контролируемом наборе и одинаковом бюджете обучить `reference_en`, `reference_pl` и `reference_en_pl_ru`. Сначала не запускать полномасштабное обучение.
5. Провести ablation reference conditioning: правильный reference, reference от другой реплики, blank/reference-ноль при одинаковых `target_mel`, noise, `t`, тексте и speaker. Ввести аналог зависимости от reference:

   `D_ref(t) = (L_reference_shuffled - L_reference_correct) / L_reference_correct`

   Считать её по зонам trajectory так же, как ранее считался `D_text(t)`. Отдельно проверить, что подмена reference меняет интонацию/ритм, но не заменяет русский текст чужим содержанием.
6. Для слухового теста на каждом кандидате сделать одинаковую матрицу: известный русский текст + reference из той же реплики, известный русский текст + reference другой реплики, unseen русский текст + reference другой реплики; для Geralt и Yennefer отдельно. Сравнивать также с baseline `step=122000`.
7. После подтверждения зависимости от reference выбрать лучший языковой режим и только затем масштабировать обучение, добавлять остальные голоса и настраивать смешивание языков.

До завершения пункта 5 полнометражное обучение трёх вариантов не запускать: сначала нужно доказать, что новая ветка действительно использует reference-аудио, а не просто продолжает копировать русский target или игнорирует дополнительную модальность.

### Experiment H — первый en-reference fine-tune от aligned 122k

Пользователь разрешил рискованный первый прогон на тестовой размерности. Запущен отдельный локальный pipeline `scripts/run_en_reference_20k.ps1`: он сначала строит английский reference-prosody cache для `manifests/phoneme_v2`, затем автоматически запускает fine-tune текущего aligned checkpoint `step=122000` ещё на `20000` шагов, до целевого `step=142000`.

Используется уже существующая в проекте prosody-модальность: английское аудио преобразуется в frame-level признаки `log-F0 / voicing / log-energy / pause`, после чего проходит через встроенный `ProsodyTransformer`. Русские phonemes и `target_speaker` остаются основными условиями, русский target mel — supervision. Для ускорения cache extraction добавлен готовый torchaudio pitch path; медленный `librosa.pyin` в этот прогон не используется. Новая prosody projection обнуляется при старте, чтобы сохранить поведение baseline до её обучения.

Артефакты этой ветки отделены от baseline: cache `cache/prosody_en_phoneme_v2/`, лог `logs/transformer_aligned_phoneme_en_refer_20k.csv`, checkpoint `checkpoints/transformer_aligned_phoneme_en_refer_20k_latest.pt`, общий pipeline log `logs/en_reference_20k_pipeline.log`. Прогон запущен в отдельном PowerShell-процессе; на момент старта выполняется extraction cache, обучение начнётся автоматически после его завершения. Оценивать результат будем тем же тестом: одинаковый русский текст и speaker при правильном, перемешанном и пустом/отсутствующем English reference.

Первичный однопроцессный extraction был остановлен до начала обучения из-за низкой утилизации CPU. Pipeline перезапущен с четырьмя независимыми worker-процессами по группам speakers; checkpoint и baseline при этом не затрагивались. На контрольной проверке после перезапуска cache строился параллельно, одновременно работали четыре Python-процесса; текущий этап остаётся только подготовкой признаков.

После исправления служебной проверки завершения worker-процессов cache успешно доведён до `25083` файлов. Fine-tune реально стартовал на `CUDA` с `step=122000`; первые записанные train-точки: `step=122050` total `0.370`, mel L1 `0.228`, CTC `0.620`, затем `step=122100` total `0.480`, mel L1 `0.325`, CTC `0.679`. Первый новый checkpoint ожидается на `step=123000`; baseline `transformer_aligned_phoneme_full_200k_latest.pt` не изменяется.

### График en-reference после 4.5k шагов

По CSV `logs/transformer_aligned_phoneme_en_refer_20k.csv` построен график `reports/plots/transformer_aligned_phoneme_en_refer_20k_training.webp`. На момент построения записаны train-точки `122050–126500` и validation на `123k`, `124k`, `125k`, `126k`. Validation total колеблется `0.512–0.532`, mel L1 — `0.282–0.301`, CTC — `1.057–1.079`; за первые ~4.5k шагов устойчивого численного улучшения ещё не видно. Это пока не вывод о качестве новой модальности: модель только начала адаптировать prosody projection.

### Контрольный тест: один русский текст, два English reference

Для текущего en-reference checkpoint `step=126000` сделан парный слуховой тест на Geralt. Русский target-текст в обоих случаях одинаковый: `Всё холоднее… Уже близко…`; speaker также одинаковый (`geralt`), seed `42`, `32` flow-steps и Vocos.

1. `ref_en_paired`: английская парная реплика той же записи — `Getting colder… Gotta be close…` (`id=0x00026947`, `2.618 s`).
2. `ref_en_other_geralt`: другая английская реплика того же Geralt — `Both were here. I'm going the right way.` (`id=0x0002a699`, `2.573 s`).

Результаты сохранены в `samples/en_refer_two_refs_geralt_158023/`. Это чистая проверка влияния reference: содержание и русский speaker-condition не менялись, менялся только English reference.

Пользователь отклонил первый «чужой» reference как слишком близкий по просодии и выбрал явный смех Geralt: `Hahaha!` (`id=0x00102085`, `1.519 s`). Второй синтез выполнен с этим reference при тех же русском тексте, speaker, seed `42`, checkpoint `step=126000`, `32` flow-steps и Vocos. Активная пара для прослушивания: `geralt_158023_ref_en_paired.wav` и `geralt_158023_ref_en_laugh_geralt.wav`. Предыдущий `ref_en_other_geralt.wav` сохранён как след промежуточного теста, но в утверждённое сравнение не входит.

### Ручное управление просодией без English reference

В `scripts/infer.py` добавлен минимальный manual-prosody режим: четыре числовых параметра задают постоянный по времени вектор `[pitch, voicing, energy, pause]`; `--target-frames` позволяет отделить длину результата от reference-аудио. На русском тексте `Всё холоднее… Уже близко…` для Geralt с checkpoint `step=126000`, seed `42`, `32` flow-steps и фиксированными `286` frames выполнены три прогона: neutral `[0,1,0,0]`, bright/high `[1.5,1,1,0]`, dark/low `[-1.5,1,-1,0]`. Все результаты имеют одинаковую длительность `3.051 s` и сохранены в `samples/manual_prosody_test_geralt_158023/`; подробности — `reports/manual_prosody_test_geralt_158023.md`. Это первый sanity-check реакции ветки на ручные признаки, а не полноценное рисование локальной интонационной кривой.

### Результат manual-prosody sanity-check

При прослушивании пользователь не услышал заметной разницы между ручными профилями. Количественно различия есть, но они слабы: neutral↔bright имеет WAV RMSE `0.0227`, корреляцию `0.938` и mel RMSE `0.0240`; neutral↔dark — WAV RMSE `0.0233`, корреляцию `0.935` и mel RMSE `0.1842`. В checkpoint `step=126000` норма `prosody_projection.weight` равна `0.655`, против `11.74` у `text_projection.weight` и `20.42` у `frame_text_projection.weight`; reference-ветка была обнулена перед en-reference fine-tune и за ~4.5k шагов ещё не набрала достаточный вес.

Дополнительная проблема самих глобальных ползунков: в обучении pitch и energy нормализуются отдельно внутри каждого reference-файла, поэтому постоянные значения `+1.5/-1.5` не соответствуют естественным обучающим траекториям. Следующий корректный ручной тест должен задавать локальные временные кривые pitch/energy и короткие участки pause при одинаковой длине, а не один константный уровень на всю фразу. До этого нельзя считать, что модель доказанно игнорирует просодию; текущий тест доказал только, что глобальные ручные сдвиги при нынешней слабой ветке почти не слышны.

### Ручные локальные кривые просодии

В `scripts/infer.py` добавлен режим `--manual-prosody-profile` с frame-level профилями `rise`, `fall` и `mid_pause`. Для одной русской фразы Geralt (`Всё холоднее… Уже близко…`) с фиксированными `286` frames, seed `42` и `32` flow-steps сгенерированы три WAV без English reference. Численный контраст rise↔fall оказался умеренным (mel RMSE `0.1045`, WAV correlation `0.9703`), тогда как профили с центральной паузой дали заметно большее спектральное отличие (mel RMSE `0.62–0.68`, WAV correlation `0.89–0.92`). Результаты сохранены в `samples/manual_prosody_curves_geralt_158023/`, отчёт — `reports/manual_prosody_curves_geralt_158023.md`. Это показывает, что модель реагирует на локальное изменение pause/voicing сильнее, чем на плавный разворот pitch/energy, но ещё не доказывает естественный перенос интонации.

Пользователь не услышал заметной разницы ни между rising/falling, ни с центральной паузой; варианты практически идентичны на слух. Эксперимент считается отрицательным для текущей reference-ветки: небольшие численные изменения mel не превращаются в управляемую просодию. Причина наиболее вероятно в objective: при en-reference fine-tune `prosody_projection` была обнулена, а отдельного prosody/reconstruction loss не было (`include_target_prosody=False`), поэтому flow-loss может решать задачу через русский текст, frame-alignment и speaker, игнорируя reference. Следующий кандидат на исправление — прямое frame-level/FiLM-подключение prosody к acoustic stream плюс явный loss на F0/voicing/energy/pause; повторять ручные ползунки на текущем checkpoint без изменения обучения смысла нет.

### Подготовлена ветка обучения с явными prosody/reference loss

Чтобы проверить гипотезу не только по косвенной реакции flow-loss, подготовлена отдельная ветка от en-reference checkpoint `step=126000`. Текущий checkpoint сохранён и не перезаписывается; обучение новой ветки пока не запускалось.

Добавлен `prosody_head`, предсказывающий для каждого acoustic frame четыре признака reference-просодии: `log-F0`, voicing, log-energy и pause. Для него добавлен прямой reconstruction objective: SmoothL1 для F0/energy, BCE для voicing/pause и опциональный delta-loss для формы траекторий pitch/energy. Метрики пишутся в обычный CSV-лог рядом с существующими `loss`, `mel_l1`, `ctc`, `duration`, `x0` и `speaker`.

Добавлен reference-ranking objective. При одинаковых `noisy_mel`, timestep, русском phoneme-тексте, speaker и `frame_text` модель сравнивает правильный prosody-reference с перемешанным reference. Loss требует, чтобы ошибка на правильном reference была ниже ошибки на чужом; одновременно записывается диагностическая зависимость `D_ref = (L_shuffled - L_correct) / L_correct`. Русский текст в этом тесте не перемешивается — меняется только reference-модальность.

В aligned-модели `frame_text` теперь можно передать извне и использовать один и тот же frame-level alignment для correct/shuffled сравнения. Новая конфигурация: `configs/transformer_aligned_phoneme_en_refer_prosody_loss.yaml`, цель — `step=146000` (20k шагов от 126k). Для старта этой ветки нужен `--init-checkpoint` от `step=126000`, а не `--resume`, поскольку добавились новые параметры `prosody_head`.

Проведены smoke-тесты на реальных batch’ах: reconstruction loss считается, `prosody_pred` имеет форму `[batch, frames, 4]`, ranking loss ненулевой и видит correct/shuffled reference (`ref_samples=4`). Новых checkpoint’ов smoke-тесты не создавали. Следующий шаг — запустить короткий прогон новой ветки и проверить, растёт ли `D_ref`/`ref_rank` и появляется ли слышимая зависимость интонации от English reference.

### Эксперимент emotion2vec на парных English/Russian репликах

Для проверки высокоуровневого expressive style скачан готовый `emotion2vec+ seed` через FunASR/ModelScope; исходные аудио не изменялись. Модель выдаёт frame-level embedding размерности `768` примерно на `50 Hz`. На его основе построен диагностический probe: официальный девятиклассовый head применён к каждому frame embedding, а рядом сохранены реальные low-level признаки `F0 / voicing / energy / pause`.

Обработано `16` пар из validation: по `8` Geralt и Yennefer, English reference и соответствующая русская target-озвучка. Артефакты находятся в `reports/emotion2vec_validation_probe/`: `summary_emotion2vec.png`, отдельные графики `pairs/*.png`, временные ряды `timeseries/*.csv`, сводка `summary.csv` и отчёт `emotion2vec_report.md`.

Первичный результат не подтверждает, что абсолютные emotion labels можно напрямую считать разметкой нашего датасета. Средний `non_neutral` probe-score равен `0.562` для English и `0.899` для Russian; русские записи непропорционально часто получают классы злости/отвращения. Это может быть языковым/доменных смещением и ограничением frame-wise применения head, который официально предназначен прежде всего для chunk/utterance classification. На одинаковых ID English и Russian labels часто расходятся.

Вывод для архитектуры: emotion2vec полезен как дополнительный high-level style-stream или auxiliary similarity loss после калибровки, но не как ground truth эмоций и не как замена F0, voicing, energy, pause, duration и speaker conditioning. Для основного `ProsodyTranslator` target-просодией остаётся акустически извлечённая русская реплика; emotion2vec можно добавить вторым, слабым условием.

#### Исправление визуализации emotion2vec

Первичная сводная картинка была неудобной: English и Russian смешивались в одних группах столбцов, а ID реплик визуально выглядели координатой X. Построены отдельные language-grid графики `en_emotions_grid.png` и `ru_emotions_grid.png`: каждая панель — один WAV, X — секунды от начала этой конкретной реплики, Y — score. Цвета классов зафиксированы одинаково для всех графиков. Отдельно добавлены `en_lowlevel_grid.png` и `ru_lowlevel_grid.png` с F0, voicing, energy и pause.

По следующему замечанию grids заменены/дополнены парными карточками `cards/*.png`: одна карточка соответствует одному ID, сверху находится русская реплика, снизу английская, обе с независимой временной осью и общей легендой цветов. Для неэмоциональных признаков добавлены аналогичные `cards_lowlevel/*.png`.

### Сравнение пользовательских записей с RU/EN/PL на emotion2vec+ base

Для проверки языковой/интонационной близости скачан готовый `iic/emotion2vec_plus_base` через FunASR/ModelScope. Из трёх одинаковых по смыслу исходных реплик Дийкстры собраны отдельные непрерывные reference-файлы: русский `59.2 с`, английский `62.6 с` и польский `59.2 с`; между исходными репликами оставлены паузы. Записи пользователя `Recording (2)` (`63.3 с`) и `Recording (3)` (`70.2 с`) конвертированы только в рабочие копии `16 kHz mono WAV`; исходные M4A не изменялись.

Для каждой записи использованы оконные base-эмбеддинги: окно `1.0 с`, шаг `0.25 с`. Сходство считалось как cosine similarity между пользовательским окном и reference-окном на том же нормированном проценте прогресса записи. Это временной диагностический ряд сходства expressive style/акустического контекста, а не процент похожести голоса и не forced alignment.

Итоговые средние значения: `Recording (2)` — RU `0.602`, PL `0.460`, EN `0.327`; по третям RU `0.598 / 0.642 / 0.565`. `Recording (3)` — RU `0.729`, PL `0.544`, EN `0.400`; по третям RU `0.732 / 0.730 / 0.724`. Таким образом, base уверенно видит наибольшую близость обеих пользовательских записей к русскому оригиналу; для `Recording (2)` на третьем участке есть заметное падение RU similarity, что согласуется с отмеченным сбиванием, но не доказывает его причину без точного word-level alignment.

Артефакты: `reports/emotion2vec_base_comparison/recording_2_similarity_timeseries.png`, `recording_3_similarity_timeseries.png`, `global_base_similarity_matrix.png`, `base_similarity_summary.csv`, `base_similarity_report.md` и кеш `base_embedding_cache/`. Скрипт повторного запуска: `scripts/compare_base_recordings.py`.

### Попарное сравнение aligned 122k: одинаковый текст против одинакового speaker

Взяты четыре сохранённых семпла `aligned_122k` из теста нестандартного текста: `Саша` и `Грека/рак`, каждый с голосом Geralt и Yennefer. На готовом `emotion2vec+ base` извлечены frame embeddings; для временного ряда использованы окна `0.40 с` с шагом `0.08 с`.

Глобальная cosine-матрица показала: одинаковый текст при смене speaker — `0.299` для «Саши» и `0.248` для «Грека»; одинаковый speaker при смене текста — `0.722` для Geralt и `0.228` для Yennefer. Среднее `speaker-change = 0.274`, среднее `text-change = 0.475`.

Это не даёт простой картины «base отделяет speaker лучше текста»: результат сильно зависит от конкретного голоса и короткой синтетической реплики. Для Geralt два текста похожи между собой, а для Yennefer — резко различаются; у обоих одинаковый текст между speaker даёт низкую similarity. Временные ряды также нестабильны: для «Саши» `0.122–0.431`, для «Грека» `-0.104–0.538`. Вывод — base-эмбеддинг на этих коротких синтетических семплах отражает не только speaker, но и артефакты/содержание/качество конкретной генерации; делать вывод о полноценном speaker disentanglement по этим четырём WAV нельзя.

Артефакты: `reports/aligned_122k_base_comparison/aligned_122k_base_pairwise_matrix.png`, `aligned_122k_base_same_text_timeseries.png`, соответствующие CSV и `aligned_122k_base_pairwise_report.md`. Скрипт: `scripts/compare_aligned_122k_base.py`.

#### Интерпретация временного ряда: base не speaker-invariant

По временным рядам пользователь отдельно отметил ключевой результат: при одном и том же тексте, но разных speaker, `emotion2vec+ base` считает речь крайне непохожей. Это видно по низкой cosine similarity Geralt↔Yennefer (`примерно 0.25–0.30`) и провалам временного ряда почти к нулю/ниже нуля.

Следовательно, base embedding содержит сильную компоненту speaker/тембра/артикуляции и чувствительность к артефактам генерации; его нельзя трактовать как чистое emotion/prosody-пространство. Это не означает, что модель различает содержание фразы как разное: метрика сравнивает акустические представления.

Практическое следствие для проекта: прямой cosine-loss между английской и русской озвучкой будет штрафовать ожидаемую смену актёра и не подходит как основной prosody-loss. Для переноса интонации нужны speaker-нормализованные признаки или отдельное сравнение формы траекторий (`F0`, energy, voicing, pause, duration), а base-embedding допустим только как слабый auxiliary сигнал после удаления/контроля speaker-компоненты.

### Аудит полноты истории на 2026-09-10

Проверена актуальная история проекта. В ней зафиксирован весь основной исследовательский путь: подготовка датасета и правильного log-mel, baseline, raw Transformer, phoneme/alignment v2, overfit-контроли, официальный VITS/MAS, full aligned Transformer, остановка на `122k`, reference/prosody-гипотеза, en-reference fine-tune, ручные prosody-тесты, emotion2vec и попарный тест speaker/text на `aligned_122k`.

Для каждого существенного этапа сохранены либо checkpoint/config/log, либо WAV/CSV/график и пользовательская слуховая оценка с решением принять/отвергнуть/продолжить. Отдельно зафиксированы критические исправления: неправильный log-mel, неверная phoneme-подача в inference, эвристическая длина вместо duration head, CPU-side forced-alignment bottleneck и то, что низкий CTC на target mel не доказывает читаемую генерацию.

Ранее не были отдельными пунктами оформлены служебные детали. Теперь зафиксировано: канонический рабочий проект находится в `C:/witchervoicesdataset/witcher_tts`; каталог OneDrive `C:/Users/miskl/OneDrive/Документы/witchervoicesdataset` на момент аудита пуст и не является источником артефактов. Окружение — Anaconda/Conda `C:/Users/miskl/.conda/envs/witcher_tts`, Python `C:/Users/miskl/.conda/envs/witcher_tts/python.exe`; CUDA использовалась там, где это отмечено в экспериментах.

Польские файлы присутствуют в исходном staging-корпусе и использованы в текущем диагностическом сравнении как `PL original`; полноценное обучение польской reference-модальности ещё не запускалось. План трёх reference-вариантов (`reference_en`, `reference_pl`, `reference_en_pl_ru`) зафиксирован выше и остаётся будущей фазой после исправления просодического objective.

Не переносились в статью дословно все короткие сообщения о ETA, загрузке CPU/GPU, повторяющиеся статусы и бытовые комментарии: они не меняют воспроизводимость или выводы. Это сознательная граница между полной экспериментальной историей и необработанным журналом чата.

### FACodec V2 / NaturalSpeech 3 на тех же синтетических семплах

По просьбе пользователя модели `emotion2vec_plus_seed` и `emotion2vec_plus_base` удалены с диска; отчёты и кеши признаков emotion2vec сохранены как исторические результаты. Освобождено примерно `2.08 GB`.

Вместо них взят официальный публичный компонент NaturalSpeech 3 — `FACodec V2` из `third_party/naturalspeech3_facodec`, с официальными весами `ns3_facodec_encoder_v2.bin` и `ns3_facodec_decoder_v2.bin` из `amphion/naturalspeech3_facodec`. Полный генеративный checkpoint NaturalSpeech 3 для локального запуска не подменялся самописной моделью: проверена именно опубликованная codec-факторизация.

На CUDA обработаны те же четыре WAV из `samples/nonstandard_compare_aligned_54k_vs_raw_128k/`:

- `aligned_122k_geralt_sasha.wav` и `aligned_122k_yennefer_sasha.wav` — один текст «Саша шла по шоссе.»;
- `aligned_122k_geralt_greka.wav` и `aligned_122k_yennefer_greka.wav` — один текст «Ехал Грека через реку, видит Грека — в реке рак.».

Для каждого файла сохранены reconstruction WAV, `spk_emb`, `prosody_feature`, VQ IDs (`q0` prosody, `q1-q2` content, `q3-q5` residual) и непрерывные групповые quantized latents. Внутри каждой текстовой пары также сделана реконструкция с переставленным speaker embedding, чтобы отдельно послушать факторизацию тембра.

Средние значения для двух пар «один текст, разные голоса» против двух пар «один голос, разные тексты»:

| сравнение | speaker cosine | raw prosody cosine | prosody latent | content latent | residual latent | q0 ID agreement | q1-q2 ID agreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| один текст, разные голоса | `0.960` | `0.926` | `0.504` | `0.798` | `0.999` | `0.012` | `0.034` |
| один голос, разные тексты | `0.969` | `0.926` | `0.115` | `0.385` | `0.999` | `0.005` | `0.000` |

Предварительный вывод: непрерывные `content latent` заметно ближе у одинакового текста (`0.798` против `0.385`), то есть FACodec на этих WAV действительно оставляет содержательную компоненту. Непрерывный prosody latent тоже различает «один текст» от «разные тексты» (`0.504` против `0.115`), но raw mel-подобные prosody features почти одинаковы (`0.926` в обоих случаях), поэтому их нельзя считать достаточным emotion/prosody-пространством. `speaker cosine` почти одинаков для всех пар (`0.960–0.969`): на этих двух коротких синтетических фразах speaker embedding не даёт чистого разделения Geralt/Yennefer.

Категориальные ID нельзя читать как обычный cosine-признак: низкое точное совпадение индексов кодбука не отменяет близость непрерывных latents. Поэтому для следующего TTS-эксперимента более информативны continuous content/prosody streams, а VQ ID оставлены как диагностический дискретный слой.

Артефакты: `reports/facodec_naturalspeech3_comparison/facodec_report.md`, `facodec_pairwise_metrics.csv`, `facodec_pairwise_heatmaps.png`, `facodec_latent_heatmaps.png`, `facodec_same_text_prosody_timeseries.png`, кеш `cache/` и reconstruction/cross-speaker WAV в `recon/`. Скрипт повторного запуска: `scripts/run_facodec_comparison.py`.

#### FACodec: разворот попарных метрик во временной ряд

Чтобы не прятать локальные различия за одним средним cosine, построен `facodec_same_text_factor_timeseries.webp`. Для каждой одинаковой фразы отдельно показаны по нормализованному времени четыре frame-level кривые Geralt↔Yennefer: raw prosody mel, continuous prosody latent, continuous content latent и continuous residual latent.

На «Саше» content latent держится в основном выше `0.7–0.9`, а prosody latent заметно проваливается на отдельных участках. На «Греке» content latent более рваный, но в среднем остаётся выше, чем при сравнении разных текстов; prosody latent имеет многочисленные локальные провалы. Residual latent почти горизонтален около `1.0`, то есть на этих WAV он не несёт различимой динамики. Новый временной ряд подтверждает: FACodec выделяет content лучше, чем просодию, а residual для текущего материала диагностически бесполезен.

Построен зеркальный временной ряд `facodec_same_speaker_factor_timeseries.webp`: теперь сравниваются разные тексты внутри одного speaker — Geralt «Саша»↔«Грека» и Yennefer «Саша»↔«Грека». В отличие от предыдущего графика, здесь speaker не меняется, меняется только содержание. Content latent заметно меняется по времени, а continuous prosody latent часто проваливается к нулю и ниже. Это дополнительно подтверждает, что текущий codec явно кодирует различия текста, тогда как просодический поток на этих синтетических WAV нестабилен.

### Plachtaa FAcodec 50k на тех же четырёх WAV

Текущая NaturalSpeech 3/FACodec V2-ветка и её результаты сохранены; ничего из неё не удалялось. Дополнительно скачан официальный checkpoint `Plachta/FAcodec` — модель авторов, обученная на заявленных `50k` часах речи и более чем `1 млн` спикеров. Официальный pipeline работает на `24 kHz`; запуск выполнен на CUDA.

На тех же `aligned_122k_geralt/yennefer_sasha/greka.wav` извлечены `timbre`, временные `prosody/content/residual` latents и дискретные коды, а также сохранены четыре codec-reconstruction WAV. Артефакты находятся в `reports/plachtaa_facodec_50k_comparison/`, повторный запуск — `scripts/run_plachtaa_facodec_50k.py`.

Средние сравнения:

| сравнение | timbre | raw prosody | prosody latent | content latent | residual latent | content IDs |
|---|---:|---:|---:|---:|---:|---:|
| один текст, разные голоса | `-0.120` | `0.748` | `0.685` | `0.675` | `0.082` | `0.031` |
| один голос, разные тексты | `0.736` | `0.637` | `0.360` | `0.434` | `0.062` | `0.010` |

По сравнению с NaturalSpeech 3/FACodec V2 Plachtaa 50k намного лучше разделяет `timbre`: одинаковый speaker даёт `0.736`, разные speaker на одном тексте — `-0.120`, тогда как у V2 было примерно `0.969` и `0.960`. Content latent у Plachtaa даёт правильный контраст (`0.675` против `0.434`), хотя слабее, чем V2 (`0.798` против `0.385`). Prosody latent также даёт различие (`0.685` против `0.360`), но это ещё не доказывает чистую speaker-independent просодию. Residual у Plachtaa не коллапсирует в единицу, как V2, но его значения низкие и пока не показывают убедительной отдельной факторизации.

Главный промежуточный результат: для нашей гипотезы о раздельных входах Plachtaa 50k выглядит более перспективным по timbre-фактору и в целом даёт более содержательно различимые латенты. Следующим шагом имеет смысл сравнить reconstruction на слух и отдельно проверить cross-speaker/voice-conversion режим; текущий запуск был именно анализом представлений.

### MSR-Codec на тех же четырёх WAV

По просьбе пользователя выполнен тот же frozen-codec факторизационный эксперимент для официального `MSR-Codec`. Использованы опубликованные mel-codec и vocoder `HerbertLI/MSR-Codec`, официальный mHuBERT-25Hz + kmeans-500 из textlesslib для semantic stream и рекомендованный авторами CAM++ для global timbre. Это не обучение новой TTS-модели: проверялась готовая много поточная кодировка и обратная реконструкция.

Входы полностью совпадают с предыдущими двумя экспериментами: `aligned_122k_geralt_sasha.wav` и `aligned_122k_yennefer_sasha.wav` с текстом «Саша шла по шоссе.», а также `aligned_122k_geralt_greka.wav` и `aligned_122k_yennefer_greka.wav` с текстом «Ехал Грека через реку, видит Грека — в реке рак.». Исходные WAV не изменялись. Для каждого файла сохранены reconstruction WAV, semantic units, global timbre, prosody/residual IDs и continuous latents; эти потоки пригодны для следующего ручного swapping-теста.

MSR-Codec действительно выдаёт четыре раздельных потока: semantic на mHuBERT units, global timbre, temporal prosody и residual detail. Средние попарные значения для «один текст, разные голоса» против «один голос, разные тексты»:

| сравнение | timbre | raw mel | semantic | prosody | residual | semantic IDs | prosody IDs | residual IDs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| один текст, разные голоса | `0.3516` | `0.7454` | `0.2527` | `0.4160` | `0.5070` | `0.2275` | `0.0000` | `0.0100` |
| один голос, разные тексты | `0.4968` | `0.4110` | `0.0325` | `0.2479` | `0.4339` | `0.0275` | `0.0100` | `0.0075` |

Главный результат: semantic stream показывает ожидаемый контраст содержания — при одинаковом тексте он заметно ближе (`0.2527` и `22.75%` ID agreement), чем при смене текста у того же speaker (`0.0325` и `2.75%`). При этом global timbre на этих коротких синтетических WAV не разделяет голоса чисто: same-text/different-speaker даёт `0.3516`, а same-speaker/different-text — `0.4968`. Поэтому MSR-Codec подтверждает наличие content-потока, но этот набор не подтверждает чистую speaker-independent факторизацию timbre/prosody.

Prosody latent ведёт себя смешанно (`0.4160` против `0.2479`), то есть на этих данных его нельзя трактовать как «эмоции в чистом виде». Residual ID agreement почти нулевой (`1.0%` и `0.75%`), а continuous residual остаётся умеренно похожим (`0.5070`/`0.4339`); его следует считать детализацией акустического сигнала. Реконструкции получены для всех четырёх файлов, с длительностью около `1.32 s`/`3.56 s` после кодековой дискретизации.

Для воспроизводимости добавлены `scripts/run_msr_codec_comparison.py` и `reports/msr_codec_comparison/`: `msr_codec_report.md`, `msr_codec_pairwise_metrics.csv`, `msr_codec_pairwise_heatmaps.webp`, `msr_codec_same_text_factor_timeseries.png`, `msr_codec_same_speaker_factor_timeseries.png`, `recon/` и `features/*.npz`. В ходе запуска понадобился небольшой Python 3.11 compatibility patch в локальной копии third-party fairseq: старые mutable dataclass defaults заменены на `default_factory`, а Hydra-регистрация научена читать factory-поля; пользовательский датасет и прошлые модели этим не затрагивались.

### Очистка MSR-Codec и выбранный формат FAcodec-кэша

После завершения сравнения удалены только пять локальных файлов весов MSR-Codec: mHuBERT, kmeans, mel-codec, vocoder и CAM++ — всего освобождено примерно `1.38 GB`. Отчёт, метрики, графики, код запуска и короткие тестовые признаки сохранены для статьи и воспроизводимости; исходный датасет и модели TTS не затронуты.

Для следующей фазы выбран нативный learned-поток Plachtaa FAcodec без ручной интерпретации его координат. В основной кэш входят только `prosody_codes` для каждой EN/PL/RU реплики, существующие пары и метаданные (`id`, язык, speaker, текст, длительность). `timbre` не сохраняется: в нашем сценарии голос задаётся русским `speaker_id`, а timbre иностранной реплики был бы мешающим фактором. `content_codes`, `residual_codes`, `raw_prosody` и continuous latents не входят в основной кэш: первые два не нужны для гипотезы переноса просодии, `raw_prosody` не является чистой разметкой просодии, а continuous latents дают основной дисковый взрыв.

Корпус EN+RU+PL содержит `209.588` часов, примерно `754516` секунд. При рабочем темпе FAcodec около `80` кадров/с один `prosody_codes` имеет один индекс на кадр. При хранении индексов как `uint16` это `120722536` байт, то есть около `0.121 GB` сырого payload (`0.112 GiB`); с NPZ-заголовками и метаданными ожидаемый фактический кэш — порядка `0.15–0.3 GB`. Для сравнения, все шесть дискретных codebook-потоков заняли бы около `0.724 GB` сырого payload, а сохранение всех continuous latent’ов по прежней оценке — около `641 GB`.

### 2026-09-10 — оптимизация построения FAcodec-кэша

Первичная оценка в несколько суток была некорректной: она попала на загрузку модели и первые тяжёлые батчи. В первой реализации на GPU оказывалась лишняя часть FAcodec, а батчи из 16 записей паддились до длительности самого длинного WAV. Обработка переведена на нативные `prosody_codes`, на GPU оставлены только encoder и prosody quantizer, а записи сгруппированы по длительности.

При лимите `8 s` на padded-длительность батча GPU использовалась не полностью. Лимит увеличен до `16 s`: уже созданные NPZ сохраняются и пропускаются при возобновлении. Текущий режим даёт примерно `1.5x` ускорение; дальнейшее увеличение опасно из-за 8-гигабайтной VRAM.

### 2026-09-10 — завершение кеша и запуск переводчика просодии

Полный кеш EN/PL/RU завершён: `64122` NPZ-файла, в каждом сохранены только три временных ряда нативных FAcodec `prosody_codes` (`uint16`). После исправления runner повторный запуск правильно пропускает готовый кеш и больше не загружает FAcodec для обучения. Исправлена ошибка запуска из неправильной рабочей директории (`ModuleNotFoundError: src`), добавлены корень проекта в рабочую директорию и `PYTHONPATH`.

`AlignedProsodyTranslator` запущен с нуля на полном корпусе, `4202144` параметров, цель — `30000` шагов. На первых `1150` шагах loss снизился примерно с `7.07` до `5.84`, `device=cuda`, среднее время шага около `0.08–0.09 s`; шаги растут. При трёх замерах GPU показывала `22–47%` загрузки и около `7.6–7.9/8.15 GB` VRAM: это не остановка, а маленькая 4M-модель при почти заполненной памятью динамической пачке. Отдельная проверка зависимости выхода от иностранной просодии обязательна после обучения.

### 2026-09-10 — завершение 30k и плато переводчика просодии

`AlignedProsodyTranslator` успешно дошёл до целевых `30000` шагов; процесс завершился штатно, чекпоинт сохранён в `checkpoints/prosody_translator_full_30k.pt`. Модель не «отказалась обучаться», но улучшение слабое и к концу прогона практически остановилось:

| метрика | шаг 1000 | шаг 30000 |
|---|---:|---:|
| validation loss | `5.6401` | `5.3704` |
| validation accuracy | `4.64%` | `6.69%` |
| validation top-5 accuracy | `14.72%` | `18.06%` |
| validation perplexity | `281.5` | `214.9` |

Относительное снижение validation loss составило около `4.8%`. Основное улучшение произошло в первой части обучения; примерно на отрезке `20k–30k` loss держался около `5.39–5.37`, а top-5 accuracy — около `18%`. Это похоже на плато, а не на полезное освоение полноценного переноса иностранной просодии.

Финальные train-значения (`loss=5.8333`, `accuracy=3.20%`, `top-5=11.50%`) не следует напрямую сравнивать с validation: train-лог — одиночный шумный minibatch, validation — усреднение по нескольким batch'ам. Следующий обязательный диагностический шаг — frozen ablation с одинаковыми `target_mel/noise/t/speaker`: правильная, перемешанная и blank иностранная просодия. Только разница `L_shuffled-L_correct` покажет, использует ли сеть иностранную просодию, а не просто учит частотный/позиционный baseline.

### 2026-09-10 — контрольный запуск модели 8M

После завершения 4M-прогона запущен отдельный контрольный эксперимент на том же полном датасете, том же готовом FAcodec-кеше, том же objective, optimizer, frame budget (`2400`), seed (`42`) и лимите `30000` шагов. Увеличена только ёмкость Transformer: `dim=224`, `ffn_dim=896`, `text_layers=3`, `prosody_layers=3`, `decoder_layers=3`; фактический размер — `7858944` обучаемых параметров.

Новый результат пишется отдельно в `logs/prosody_translator_full_8m_30k.csv` и `checkpoints/prosody_translator_full_8m_30k.pt`; чекпоинт 4M сохранён. На шаге `1000` зафиксировано: train loss `5.9086`, accuracy `3.52%`, top-5 `12.13%`, средняя скорость около `0.079 s/step`, устройство `cuda`. Процесс запущен detached, чтобы закрытие приложения его не остановило.

### 2026-09-10 — завершение 8M и frozen prosody ablation

8M-модель штатно завершила `30000` шагов. Финальный validation: loss `5.3643`, accuracy `6.76%`, top-5 `18.16%`. В сравнении с 4M (`5.3704`, `6.69%`, `18.06%`) выигрыш минимален, поэтому увеличение размера не является главным решением текущей проблемы.

Выполнен frozen ablation на одинаковых `16` validation-batch’ах (`34547` валидных кодов на модель). Для каждой модели сохранялись исходные тексты и target-коды; менялся только входной иностранный prosody sequence:

| модель | correct loss | shuffled loss | blank loss | D_shuffled | D_blank |
|---|---:|---:|---:|---:|---:|
| 4M | `5.37110` | `5.38914` | `5.38423` | `0.336%` | `0.244%` |
| 8M | `5.36472` | `5.38266` | `5.44153` | `0.334%` | `1.432%` |

Вывод: 8M немного использует наличие входного prosody-потока, но при перемешивании подробного временного ряда loss меняется всего на `0.33%`. Это не похоже на полноценный перенос просодии; скорее сеть использует грубый статистический сигнал, длину или распределение кодов. У 4M зависимости от условия практически не обнаружено. Результат сохранён в `reports/prosody_translator_ablation_4m_vs_8m.json`; обучение остановлено только штатным завершением, новые модели не запускались.

### 2026-09-10 — fine-tune aligned TTS на native русской просодии

Следующая проверка гипотезы выполняется не с нуля. В качестве инициализации выбран чистый aligned phoneme-чекпойнт `checkpoints/transformer_aligned_phoneme_full_200k_latest.pt`, фактически сохранённый на шаге `122000`. Он обучался на русском фонемном тексте и обычном flow/CTC/duration/speaker objective, без использования просодийного входа. Предыдущая ветка `en-refer` и другие чекпойнты, где уже обучалась просодия, намеренно исключены: они считаются загрязнёнными для этого сравнения.

Вход просодии подключён из готового кеша `cache/facodec_prosody_codes_full`: используются нативные временные `ru`-коды FAcodec (`uint16`), без ручной интерпретации координат. Для каждого примера кодовая последовательность проходит через embedding + prosody Transformer adapter; padding передаётся отдельной маской. В `train_stage_a.py` оставлен существующий reference-ranking loss: при одинаковых `target_mel/noise/t/speaker/text` правильный prosody-код должен давать меньшую flow-ошибку, чем shuffled-код. Это нужно, чтобы проверить реальное использование просодии, а не только её загрузку. Старый 4-мерный continuous-prosody loss в этой ветке выключен.

Из `25083` записей манифеста в fine-tune вошли `25082`: единственная запись без полного multilingual FAcodec-кеша (`0x0010ba29`, `Хм.`) исключена только из производной выборки; исходный датасет не менялся. Цель — `20000` fine-tune-шагов с отдельными логом и чекпойнтом:

- конфиг: `configs/transformer_aligned_phoneme_ru_native_prosody_20k.yaml`;
- запуск: `scripts/run_ru_native_prosody_finetune.ps1`;
- чекпойнт: `checkpoints/transformer_aligned_phoneme_ru_native_prosody_20k_latest.pt`;
- CSV: `logs/transformer_aligned_phoneme_ru_native_prosody_20k.csv`.

Перед длинным запуском выполнен CUDA-smoke на одном шаге: загрузка `316/319` совместимых параметров чистой модели, train/validation forward и `ref_rank` прошли успешно. Три несовместимых параметра относятся к новому входному code-embedding и ожидаемо инициализируются заново; временный smoke-чекпойнт не является результатом эксперимента.

Длинный fine-tune запущен detached с `--manifest-dir manifests/phoneme_v2`; на первом контрольном чтении достигнут шаг `450/20000`. По пяти замерам `nvidia-smi dmon` средняя загрузка SM составила около `72%`, занятая VRAM — `3.64/8.15 GB`; процесс использует CUDA и не требует открытого приложения.

Первый запуск остановился на шаге `900` из-за `CUDA out of memory` внутри DataLoader `pin_memory`-потока, а не внутри forward/backward модели. Чекпойнт не успел сохраниться (первый save был на 1000), поэтому модельный результат этого запуска не используется; CSV и stdout сохранены под суффиксом `_pin_oom` для истории. Для повторного запуска снижены `num_workers: 4→2`, `prefetch_factor: 2→1`, отключён `pin_memory`, а сохранение выставлено каждые `500` шагов.

Повторный запуск выполнен с чистого `122k` init-чекпойнта. На контрольном чтении он стабильно достиг `150/20000`; за 25 секунд обработано около `100` шагов, без повторения OOM. После отключения pinning наблюдаемая загрузка SM по пяти замерам составила около `66%`, VRAM занято `3.64/8.15 GB`; текущая оценка полного прогона — около `80–90 минут`.

На шаге `2000` сохранён отдельный чекпойнт. Ранний `D_ref` слабый, но не нулевой: при взвешивании по числу eligible-сэмплов он составил `0.10%` на шагах `1–500`, `0.30%` на `501–1000`, `1.46%` на `1001–1500` и `1.76%` на `1501–2000`. Эти значения нельзя сравнивать с `5%` как с доказанным порогом качества: `0.05` — лишь внутренний hyperparameter `margin` в ranking loss. Полноценное использование просодии ещё не доказано; однако ветка начала получать различие между correct и shuffled reference. Следующая диагностическая граница — `4–5k`, после неё принимать решение о продолжении до `20k`.

По решению пользователя горизонт увеличен до `50000` шагов. После остановки на сохранённом `step=2000` runner перезапущен с `--resume`, поэтому обученные 2000 шагов и новая prosody-ветка сохранены. Scheduler корректно создан заново с горизонтом `50000`; целевой чекпойнт и CSV теперь отдельные: `checkpoints/transformer_aligned_phoneme_ru_native_prosody_50k_latest.pt` и `logs/transformer_aligned_phoneme_ru_native_prosody_50k.csv`. На первом шаге продолжения (`2050`) процесс работает на CUDA.

Fine-tune достиг последнего train-лога `4350` и validation `4000`, после чего завершился аварийно с `CUDA out of memory` внутри reference-ranking forward. Сохранённый checkpoint остаётся на `step=4000`; лимит `50000` не достигнут. В отчёте по переводу просодии этот запуск считается диагностически незавершённым.

### 2026-09-10 — восстановление после OOM и detached-перезапуск

Причина остановки подтверждена: reference-ranking делал второй forward на полном eligible-батче, что на 8-ГБ GPU иногда уходило в shared GPU memory и завершалось `CUDA out of memory`. Чекпойнт `transformer_aligned_phoneme_ru_native_prosody_50k_latest.pt` был проверен загрузкой и содержал корректный `step=4000`.

Для безопасного продолжения reference-ветка ограничена четырьмя равномерно выбранными eligible-сэмплами (`reference_conditioning_loss.max_samples=4`). При этом все связанные входы — русский текст, text mask, speaker, noisy mel, timestep, target, mel mask, frame alignment и shuffled native prosody — выбираются одним и тем же `selected_indices`; исправлена ошибка прежнего rescue-патча, где текст оставался размером полного eligible-батча. Настройка DataLoader сохранена (`num_workers=2`, `prefetch_factor=1`, `pin_memory=false`).

Одношаговый smoke-run с resume от `step=4000` прошёл до `4001` без OOM и shape mismatch. Затем основной runner запущен через явный PowerShell 7 в detached-режиме; запуск через Windows PowerShell 5 был отброшен, так как он не поддерживает синтаксис runner. На первом контрольном участке после восстановления:

- `step=4050–4550` уже записаны в отдельный resume-CSV;
- сохранён новый основной checkpoint `step=4500`;
- Python-процесс остаётся живым независимо от окна Codex;
- `nvidia-smi`: около `84%` GPU, `3.33/8.15 GB` VRAM, `79°C`.

Обучение продолжается к цели `50000`; исходные датасет и чистый init-чекпойнт не изменялись.

После продолжения скорость составила около `200 шагов за 25 секунд` (`≈0.125 s/step`); текущий прогресс — `2250/50000`, ориентировочный остаток — около `1 ч 40 мин` плюс время периодических validation/save. Средняя загрузка SM по пяти замерам — около `72%`, VRAM `3.65/8.15 GB`, температура GPU `76–81°C`.

### 2026-09-10 — остановка на 9000 и native-prosody inference grid

По запросу fine-tune остановлен на целевом `step=9000`, без продолжения к `50000`. При остановке train-строка уже была записана, но checkpoint ещё не успел сохраниться; поэтому безопасно выполнены дополнительные `500` шагов от сохранённого `8500` до `9000` отдельным коротким запуском. Checkpoint проверен загрузкой: `step=9000`.

Финальная validation на `9000`: `loss=0.60703`, `mel_l1=0.29291`, `ctc=1.06466`, `ref_correct=0.22373`, `ref_shuffled=0.28074`, `D_ref=26.05%`, `ref_samples=3.875`. Это подтверждает влияние native reference на flow prediction, но не является процентом перенесённой интонации.

Для слуховой проверки создан `samples/native_prosody_9000_grid_geralt/` на одной validation-реплике Geralt (`0x00028d16`, текст «Я ищу одну женщину…»), одном speaker ID, одинаковых `target_frames`, `flow_steps=12` и seed `42`. Варианты: `geralt_correct.wav` с собственной RU-просодией, `geralt_shuffled.wav` с RU-просодией другой реплики Geralt (`0x0002942e`), `geralt_blank.wav` без reference; рядом сохранён `target_original.wav` и `metadata.json`. Исходные аудио и датасет не менялись.

### 2026-09-10 — контрастная native-prosody проверка

Чтобы убрать confound длины reference, сделан второй grid с тем же checkpoint `step=9000`, speaker Geralt, seed `42`, русским target-текстом «Формалин. Сомневаюсь, что Фабиан его использовал в работе…» и одинаковым target duration `4.13 с`. Correct reference — собственная реплика `0x0004f984` (`4.14 с`, `331` кодовый кадр); shuffled reference — явный смех `0x00102085`, «Ха-ха-ха!» (`4.10 с`, `328` кодовых кадров). Вариант blank снова не содержит prosody memory.

Результат сохранён в `samples/native_prosody_9000_grid_geralt_contrast/` как `target_original.wav`, `geralt_correct.wav`, `geralt_shuffled.wav`, `geralt_blank.wav` и `metadata.json`. Обучение и исходные данные не менялись.

Слуховой результат контрастной проверки: `correct` субъективно почти совпал с `blank`; отдельной слышимой интонации target-reference не появилось. `shuffled` заметно изменил манеру речи — ощущение «говорит с трибуны» и выраженный грузинский акцент. Следовательно, native reference-ветка действительно влияет на генерацию, но в текущем виде извлекает/впрыскивает побочный глобальный стиль или артикуляционный шаблон, а не корректно переносит целевую локальную просодию. Validation `D_ref=26.05%` трактуется только как влияние reference на flow-loss, не как процент успешного переноса интонации.

### 2026-09-10 — дополнительные reference-просодии для локальной ветки

По запросу добавлены два новых `shuffled` reference без изменения target-текста, speaker, seed, длительности и flow steps:

- `0x0006d92c`, Geralt: «Может, ты и прав… Ладно, возьму эту. Надеюсь, ты не ошибся.» — в WAV обнаружена внутренняя низкоэнергетическая пауза примерно `2.5 с`;
- `0x000286fa`, Yennefer: «И почему я не удивлена? Геральт, Геральт, что ты надел?» — seen-текст Йеннифэр, использован именно как prosody reference, не как входной текст.

Сгенерированы отдельные grids `samples/local_prosody_9000_grid_geralt_long_pause/` и `samples/local_prosody_9000_grid_geralt_yennefer_seen/`, каждый содержит `correct`, `shuffled`, `blank`, исходную target-реплику и `metadata.json`. Исходные аудио и чекпойнты не изменялись.

Слуховое наблюдение по дополнительным reference: пока слышна только деградация речевых функций и занижение громкости. Перенос длинной паузы или узнаваемого интонационного рисунка не выявлен. Это усиливает вывод, что текущая локальная ветка использует FAcodec-коды как нестабильный побочный акустический сигнал, а не как управляемую просодию.

Рабочая гипотеза: проблема может быть прежде всего в самом представлении `FAcodec prosody_codes`. Эти дискретные коды, вероятно, не являются чистым временным рядом просодии и могут смешивать энергию, артикуляционные/фонетические признаки, тембр и другие акустические остатки. При переносе между голосами это проявляется как потеря громкости и деградация речи, а не как перенос интонации. Гипотеза требует отдельной проверки, но продолжать обучение этой ветки вслепую пока нецелесообразно.

### 2026-09-10 — inference grid локальной native-prosody ветки

Повторён тот же контрастный слуховой эксперимент, что и для предыдущей global-memory native ветки: чекпойнт `transformer_aligned_phoneme_ru_native_prosody_local_9k_latest.pt` (`step=9000`), один и тот же русский текст Geralt, одинаковые `target_frames=388`, `flow_steps=12` и `seed=42`. Менялся только входной reference:

- `correct` — собственная русская просодия target-записи `0x0004f984`;
- `shuffled` — русская просодия другой записи `0x00102085`, «Ха-ха-ха!»;
- `blank` — без prosody reference.

Результаты сохранены отдельно в `samples/local_prosody_9000_grid_geralt_contrast/`: три WAV, `target_original.wav`, mel-массивы и `metadata.json`. Исходный датасет, чистый чекпойнт 122k и чекпойнты обеих веток не изменялись. Это прямое сравнение с `samples/native_prosody_9000_grid_geralt_contrast/`; слуховой вывод должен делаться по WAV, а не по одному `D_ref`.

Слуховая оценка пользователя: `shuffled` стал заметно тише, «бухим», с плохо связывающимися словами; `blank` говорит очень тихо. Наблюдаемая разница в основном относится к громкости, стабильности артикуляции и общему состоянию голоса. Это подтверждает, что локальная reference-ветка активна, но пока не доказывает перенос временной интонации: высокий `D_ref=86.40%` отражает чувствительность flow-loss к reference, а не качество просодийного преобразования. На текущем этапе вероятнее всего код используется как грубый глобальный акустический/энергетический сигнал, тогда как локальный перенос F0, пауз и акцентных контуров ещё не выделен.

### 2026-09-10 — Frankenstein prosody v1 и удаление FAcodec

По запросу удалены только FAcodec-веса и производные кеши: Hugging Face `models--amphion--naturalspeech3_facodec` (`0.42 GB`), проектный `cache/facodec_prosody_codes_full` (`0.13 GB`) и `reports/plachtaa_facodec_50k_comparison/model_cache` (`1.40 GB`). Исходные WAV, датасет и third-party исходный код не изменялись.

Собран явный диагностический кеш `cache/prosody_frankenstein_v1` на готовых openSMILE ComParE/eGeMAPS и Praat через Parselmouth. Для каждого 10-мс кадра сохранены F0, относительный pitch, voicing, RMS/relative energy, loudness, spectral tilt, pause score и независимые Praat F0/intensity. В кеш вошли «Всё холоднее…» Geralt, «И почему я не удивлена? Геральт, Геральт, что ты надел?» Yennefer, длинная реплика Дийкстры `0x00104ad0` в RU/EN/PL и синтетические Sasha/Greka для Geralt/Yennefer. Pitch-выбросы очищаются robust-фильтром; фонемные длительности не выдумываются без MFA forced alignment.

PNG и числовой отчёт находятся в `reports/prosody_frankenstein_v1/`. Сравнение Sasha/Greka показывает сильную голосозависимость pitch-контура: нормированный F0 correlation Geralt/Yennefer около `-0.116` для Sasha и `0.073` для Greka; energy-контуры заметно ближе (`0.892` и `0.800`), но остаются разными по амплитуде. Это диагностический результат по синтетическим WAV, не доказательство независимости просодии от speaker.

MFA `3.4.2` установлен в рабочую Python-среду, но его runtime не запускается из-за отсутствующего бинарного `_kalpy` (в штатной установке он приходит через conda-forge). Поэтому Frankenstein v1 не подменяет phone-level timing самодельной оценкой; для него сохранены только честные кадрочные признаки и pause score.

### 2026-09-10 — чистый эксперимент с локальным temporal prosody injection

После слухового вывода `correct≈blank`, а `shuffled` меняет глобальную манеру, предыдущая native-reference ветка не продолжается. Создана отдельная архитектура `transformer_flow_aligned_local_prosody`: FAcodec `ru`-коды сначала проходят через code Transformer, затем каждая валидная кодовая последовательность линейно ресэмплируется на валидную mel-временную ось именно своего аудио и добавляется локально к акустическому скрытому состоянию соответствующего кадра. Prosody не добавляется в общий text/style memory.

Новая ветка стартует чистым `transformer_aligned_phoneme_full_200k_latest.pt` (metadata `step=122000`), а не загрязнённым checkpoint прежнего 9k fine-tune. Локальная prosody projection zero-initialized, поэтому на первом шаге базовая 122k-модель сохраняет исходное поведение; новые prosody-кодовые параметры обучаются отдельно. Горизонт для честного сравнения с прежним экспериментом — `9000` шагов.

Файлы новой ветки: `configs/transformer_aligned_phoneme_ru_native_prosody_local_9k.yaml`, `scripts/run_ru_native_prosody_local_9k.ps1`, `checkpoints/transformer_aligned_phoneme_ru_native_prosody_local_9k_latest.pt`, `logs/transformer_aligned_phoneme_ru_native_prosody_local_9k.csv`. Старый checkpoint `transformer_aligned_phoneme_ru_native_prosody_50k_latest.pt` и исходные данные не перезаписываются.

Запуск завершён штатно на `step=9000`; checkpoint проверен загрузкой и содержит архитектуру `transformer_flow_aligned_local_prosody`, `321` tensor entries. Финальная validation: `loss=0.53254`, `mel_l1=0.30041`, `ctc=1.04507`, `ref_correct=0.20567`, `ref_shuffled=0.37391`, `D_ref=86.40%`, `ref_samples=4.0`. В сравнении с прежней global-memory native веткой на 9k (`D_ref=26.05%`) локальная подача даёт существенно более сильную зависимость flow prediction от reference, но это ещё не эквивалент слышимого переноса просодии: следующий шаг — одинаковая inference-сетка `correct/shuffled/blank` и прослушивание.

### 2026-09-10 — рабочее окружение MFA и `_kalpy`

По запросу устранена проблема с отсутствующим бинарным модулем `_kalpy`. Pip-установка MFA в основной среде оставлена как есть, но для рабочего запуска создано изолированное conda-совместимое окружение:

- Miniforge: `C:\Users\miskl\miniforge3`;
- окружение: `C:\Users\miskl\miniforge3\envs\mfa`;
- Montreal Forced Aligner `3.4.2` из `conda-forge`;
- в окружение входят `kalpy`, `kaldi`, `pynini`, `praatio` и нативный `_kalpy`;
- скачаны готовые модели и словари `russian_mfa`, `english_mfa`, `polish_mfa`.

Проверка завершена успешно: `import _kalpy, kalpy, montreal_forced_aligner` и `mfa --help` возвращают код `0`. При прямом запуске Python DLL не находились из-за Windows PATH; запуск с путями `envs\\mfa\\Library\\bin`/`Scripts` исправляет это. Исходные WAV, датасет, чекпойнты и существующий Frankenstein-кеш не изменялись. После установки свободно около `20.8 ГБ`.

### 2026-09-10 — MFA alignment и PNG-визуализация Frankenstein v1

MFA прогнан на девяти контрольных аудио: ключевые русские реплики Geralt/Yennefer/Dijkstra, длинная реплика Dijkstra на RU/EN/PL и синтетические «Саша»/«Грека» для Geralt/Yennefer. Все три языка завершились с кодом `0`. CSV с границами слов и фонем сохранены в `cache/mfa_frankenstein_v1/aligned/`; исходные записи не перезаписывались.

Созданы PNG с общей временной осью: energy, очищенный F0 и phone/word intervals. Они лежат в `reports/prosody_frankenstein_v1/mfa_alignment/`. Серые области обозначают интервалы вне слов/паузы; нижняя дорожка — реальные MFA phone boundaries, а не равномерно придуманные интервалы.

### 2026-09-10 — Frankenstein утверждён как официальный prosody extractor

Зафиксирован состав model-ready prosody-представления для следующей фазы:

- оставить очищенный относительный `log-F0`/F0 в полутонах, нормированный относительно собственного голоса, плюс `voicing_probability` и `f0_valid`;
- оставить относительную energy в dB, нормированную внутри реплики;
- добавить MFA phone/word durations, интервалы слов и паузы между ними;
- хранить `pause_mask`/speech mask как отдельный временной канал, а не смешивать паузы с F0;
- `time_sec` оставить только метаданными временной оси.

В model-ready слой не включать сырые `f0_hz`, `praat_f0_hz`, `energy_dbfs`, `praat_intensity_db`, дублирующий `loudness` и `spectral_tilt_alpha_ratio`: они либо голосо-/записывающе-зависимы, либо дублируют канонические признаки. Сырой v1-кеш не удалять до проверки новой схемы — он остаётся audit-слоем для разбора ошибок pitch tracker и MFA.

F0 намеренно сохранён: без него переводчик видит только ритм, громкость и паузы, но теряет мелодику фразы, вопросительность, ударные подъёмы/спады и часть эмоциональной структуры. В модель идёт не сырой Hz-трек, а robust-очищенный относительный контур с явной маской валидности.

### 2026-09-10 — контрольный checkpoint удалён, запущен per-channel prosody-loss эксперимент

По запросу остановлен safe-run и удалён только его производный checkpoint `checkpoints/transformer_aligned_phoneme_frankenstein_ru_30k_latest.pt`. Кеш `cache/prosody_frankenstein_ru_model_v1`, чистая база `transformer_aligned_phoneme_full_200k_latest.pt`, логи и отчёты сохранены.

Запущен новый чистый run `transformer_aligned_phoneme_frankenstein_ru_prosodyloss_30k` от той же 122k-базы. Включён существующий в коде auxiliary objective с отдельными компонентами: F0 (`smooth_l1`), voicing (`BCE`), energy (`smooth_l1`), pause (`BCE`) и temporal delta для F0/energy. Общий auxiliary weight установлен `0.10`; основной flow/x0/CTC/duration/speaker objective не заменён. На шаге 1: `main_loss=0.42153`, `prosody_loss=0.16373`; на шаге 50: `loss=0.43215`, `prosody_loss=0.05972`, включая `F0=0.04094`, `voicing=0.53028`, `energy=0.08436`, `pause=0.19766`, `delta=0.03631`. Это первый запуск, где соответствие каждому просодийному параметру измеряется напрямую.

### 2026-09-10 — первый model-ready Frankenstein cache и fine-tune основной TTS

Для проверки, может ли основная aligned TTS использовать явный временной prosody-вход, построен отдельный русский кеш `cache/prosody_frankenstein_ru_model_v1`. Он содержит `25 083/25 083` записей из `manifests/phoneme_v2`, по пути `speaker/id.npz`. Канонический массив `prosody` имеет форму `(mel_frames, 4)` и `float32`: relative F0 в полутонах, voicing probability, relative energy и pause score. В каждом файле также оставлены audit-массивы; битых/частично записанных файлов нет. Размер кеша около `0.218 ГБ`. Исходные WAV, staging-датасет, старый `prosody_frankenstein_v1` и старые чекпойнты не изменялись.

Запущен отдельный fine-tune `transformer_aligned_phoneme_frankenstein_ru_30k` на полном `phoneme_v2`. Инициализация — чистый `checkpoints/transformer_aligned_phoneme_full_200k_latest.pt` с metadata `step=122000`; старые FAcodec/native-prosody checkpoint не использовались. Совместимо загружено `318/320` tensor entries, отсутствуют только параметры новой prosody-ветки. `prosody_projection` обнулена, чтобы исходная 122k-модель сохраняла исходное поведение на старте, а способность использовать Frankenstein-вход осваивалась с нуля. Основная задача остаётся flow/x0/CTC/duration/speaker, без искусственного prosody reconstruction loss.

Первые `1000` шагов выполнены и сохранены; validation на шаге 1000: `loss=0.52114`, `mel_l1=0.28868`, `ctc=1.06316`. Попытка продолжить с `frame_budget=10000` и `num_workers=8` подняла GPU-загрузку, но завершилась CUDA OOM на шаге `1300`; checkpoint 1000 остался целым. Рабочий run возобновлён из него с безопасным `frame_budget=5000`, `num_workers=4` и `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (на данной сборке параметр не поддерживается, поэтому полагаться на него нельзя). В safe-режиме GPU достигает `~97%`, VRAM держится около `2.5/8.1 ГБ`, OOM не наблюдается. На момент последней проверки run был на шаге `3300/30000`; логи: `logs/transformer_aligned_phoneme_frankenstein_ru_30k_safe5k.csv` и `.log`.

### 2026-09-10 — остановка per-channel prosody-loss run и listen grid

По запросу остановлено обучение `transformer_aligned_phoneme_frankenstein_ru_prosodyloss_30k`. Дочерний Python-процесс также завершён; GPU-процессов обучения после остановки не осталось. Последний сохранённый checkpoint — `checkpoints/transformer_aligned_phoneme_frankenstein_ru_prosodyloss_30k_latest.pt`, `step=18500` (обучение не продолжалось после остановки).

Для слуховой проверки сгенерирован `samples/prosodyloss_stop_step18500_listen/`: для Geralt и Yennefer по три условия (`correct`, `shuffled`, `blank`) при одинаковом seed `42`, 32 flow-steps и фиксированной длине каждой контрольной реплики. Рядом с каждым набором сохранена копия исходной русской реплики; staging-исходники не изменялись. Полная схема и параметры записаны в `samples/prosodyloss_stop_step18500_listen/metadata.json`.

### 2026-09-10 — карикатурная синтетическая просодия, inference-only

Чтобы отделить незаметное влияние просодии от полного игнорирования входа, без переобучения сгенерирована экстремальная контрольная сетка на том же checkpoint `step=18500`. Варианты: `neutral`, `cartoon_bounce` (резкие скачки F0/энергии и короткие искусственные «икоты»), `cartoon_question` (театральная пауза и сильный подъём к концу) и `cartoon_staccato` (повторяющиеся провалы voicing/pause). Для каждой из двух контрольных реплик использованы одинаковые seed `42`, 32 flow-steps и фиксированная длина; исходные WAV скопированы рядом. Результаты: `samples/prosodyloss_stop_step18500_cartoon_prosody/`. Модель, чекпоинт и исходные записи не изменялись.

Слуховая гипотеза пользователя: у карикатурных вариантов слегка смещается акцентное слово и энергия, но явного следования просодии нет. Контроль с одинаковым seed подтверждает, что эффект реальный, а не только субъективный: RMS-отклонение извлечённых признаков от `neutral` для Geralt составило примерно `0.48–0.61` по F0 и `0.13–0.15` по energy; для Yennefer `0.64–0.84` по F0 и `0.42–0.52` по energy. Однако корреляция выходного F0/energy с заданной карикатурной кривой осталась слабой или нестабильной. Следовательно, prosody-вход не игнорируется полностью, но текущая модель использует его скорее как слабое акустическое смещение (в том числе локальные изменения растяжения/акцента), а не как управляемый временной ряд. Изменение длительности не является отдельным контролируемым каналом: число mel-кадров фиксируется заранее, поэтому «растяжение» не доказывает, что модель освоила prosody-to-duration.

### 2026-09-10 — продолжение остановлено пользователем на step 22000

Запуск был продолжен корректно через `--resume` с checkpoint `step=18500`, но по запросу пользователя остановлен на validation `step=22000`, до исходного target `30000`. Последний checkpoint целый и содержит `step=22000`; в CSV дополнительно записан train sample `step=22050`. После остановки Python/GPU-процессов обучения не осталось.

По фактическому CSV построены статические PNG: `reports/training_losses/transformer_aligned_phoneme_frankenstein_ru_prosodyloss_30k_partial_22000/main_losses.png` и `prosody_losses.png`. На графиках train показан сырым бледным рядом и сглаженной кривой, validation — отдельными точками; пунктиром отмечены resume `18500` и недостигнутый target `30000`.

### 2026-09-10 — per-channel run окончательно остановлен на плато

По решению пользователя дальнейшее доведение до `30000` отменено: после возобновления с `22000` run дошёл до train `25000`, последний checkpoint сохранён на `24500`, после чего процесс остановлен. Validation на `24500`: общий loss `0.60035`, prosody-loss `0.08258`; устойчивого улучшения просодийной зависимости относительно ранних точек не наблюдается.

PNG обновлены по полному фактически записанному CSV в `reports/training_losses/transformer_aligned_phoneme_frankenstein_ru_prosodyloss_30k_partial_25000/`: `main_losses.png` и `prosody_losses.png`. Target `30000` на графике оставлен как недостигнутая контрольная отметка.

### 2026-09-11 — ranking-loss продолжение и контрольная listen-grid

Запущен отдельный ranking-loss run от сохраненного checkpoint per-channel этапа `step=24500`: `transformer_aligned_phoneme_frankenstein_ru_rankloss_10k`. Архитектура и датасет не менялись; добавлен существующий в коде reference-ranking objective, который сравнивает correct и shuffled Frankenstein-prosody при одинаковых target/noise/timestep/speaker/text. Целевой абсолютный шаг — `34500`.

Run завершён штатно на `step=34500`, checkpoint `checkpoints/transformer_aligned_phoneme_frankenstein_ru_rankloss_10k_latest.pt` сохранён; после завершения Python/GPU-процессов не осталось. Финальная validation: `ref_dependency=0.7635`, `ref_correct=0.2442`, `ref_shuffled=0.4144`, `ref_rank=0.7545`, `ref_samples=4`. Это подтверждает, что при одинаковом акустическом состоянии модель сильнее различает правильную и перемешанную просодию, но не является само по себе доказательством полноценного слухового переноса интонации.

Для слуховой проверки без дальнейшего обучения создана сетка `samples/rankloss_34500_reference_grid/`: Geralt и Yennefer, один seed `42`, `32` flow-steps, фиксированная длина, условия `correct/shuffled/blank` и копии исходных реплик. Входные тексты: «Всё холоднее… Уже близко…» и «И почему я не удивлена? Геральт, Геральт, что ты надел?». Исходные WAV и датасет не изменялись.

### 2026-09-11 — карикатурный prosody-тест финального ranking-loss checkpoint

Без переобучения, на checkpoint `step=34500`, запущена готовая inference-only сетка `samples/rankloss_34500_cartoon_prosody/`. Для Geralt и Yennefer при одинаковых `seed=42`, `32` flow-steps и фиксированной длине сгенерированы `neutral`, `cartoon_bounce` (резкие скачки F0/energy и искусственные провалы voicing), `cartoon_question` (театральная пауза и экстремальный подъём) и `cartoon_staccato` (повторяющиеся паузы/обрывы). Рядом сохранены исходные реплики и `metadata.json`; модель и исходные данные не изменялись.

Слуховой результат: отчётливо слышна разница между `neutral` и `cartoon_bounce` главным образом по энергии/манере подачи. `cartoon_question` и `cartoon_staccato` звучат промежуточно и не дают чистого ожидаемого эффекта. Это подтверждает реальное, но частичное использование prosody-входа: модель меняет интонационную окраску, однако сильнее сохраняет выученный акустический паттерн конкретной фразы. На графиках наиболее выраженная систематическая динамика наблюдается в `ref_dependency`; основной flow, CTC и поканальные prosody-loss остаются около плато. Следующий архитектурный вопрос — не продолжать этот objective вслепую, а усилить локальную покадровую привязку prosody к акустическому hidden-state и отдельно проверить длительности/паузы.

### 2026-09-11 — карикатурный тест управления длительностью фонем

Проверено, использует ли inference текущей aligned-модели выученную информацию о длительности отдельных фонем. В обучении длительность действительно присутствует: CTC forced alignment на целевом mel даёт покадровые phone durations, duration head оптимизируется против них, а при inference его предсказания распределяются по фонемам и нормируются под общий `target_frames`.

Создан inference-only тест `samples/rankloss_34500_duration_cartoon/` на checkpoint `step=34500`: `auto_duration`, растяжение первой ударной гласной в `6x`, растяжение всех ударных гласных в `4x` и общая длина `1.5x`. В вариантах с отдельными фонемами общая длина оставлена прежней, поэтому выбранные звуки растягиваются за счёт сокращения остальных; global-вариант изолирует эффект общей длительности. Для Geralt первой фонемой теста стала `ˈɵ`, для Yennefer — `ˈu`. Исходники, checkpoint и датасет не изменялись.

Уточнение по MFA-графику `reports/prosody_frankenstein_v1/mfa_alignment/geralt_colder_ru.png`: он использован как диагностическая визуализация реального F0/energy и MFA word/phone boundaries, но сам PNG и MFA phone intervals не подавались в текущий rank-loss checkpoint. В model-ready кеше `cache/prosody_frankenstein_ru_model_v1` для обучения лежат только четыре mel-frame канала: relative F0, voicing, relative energy и pause score. Длительность в текущей aligned TTS приходит из отдельного training-time CTC forced alignment по target mel: его `alignment_durations` используются как target для `duration_head`. Поэтому недавний duration-тест проверял выученный duration head и ручное изменение frame-to-phone expansion, а не прямое использование MFA-разметки. Прямой MFA-duration контроль остаётся отдельным возможным экспериментом.

### 2026-09-11 — прямой inference-тест с MFA-длительностями

Проведён отдельный inference-only тест `samples/rankloss_34500_mfa_duration/` на финальном checkpoint `step=34500`. Реальные интервалы MFA из контрольных графиков сопоставлены с текущей последовательностью русских фонем по словам; при несовпадении phone inventory длительности интерполированы внутри слова, а паузы между словами сохранены как длительность `<wb>`. Сравнены `auto_duration`, `mfa_duration`, `mfa_first_stressed_4x` и `mfa_all_stressed_3x` для Geralt и Yennefer при одинаковых `seed=42`, `32` flow-steps и общей длине реплики. Checkpoint, датасет и исходные WAV не изменялись.

Слуховой результат пользователя: эффект MFA-длительностей значимый и хорошо слышен, но при сильном растяжении немного ухудшаются произношение и разборчивость. Интерпретация: модель действительно реагирует на frame-to-phone duration control, однако прямое ручное изменение выводит её за обученное распределение длительностей; акустический decoder не был обучен сохранять качество при экстремальной переразметке. Это подтверждает ценность MFA-duration как управляющего сигнала, но не означает, что его нужно просто подставлять в inference без адаптации.

### 2026-09-11 — решение по сохранению базового duration prior

Базовый `duration_head` текущей модели не заменяется MFA-таймингами. Он уже выучил естественное распределение длительности фонем из текста, speaker и CTC alignment, поэтому остаётся основным prior. MFA подключается как дополнительная корректировка: новый duration adapter получает базовое предсказание и внешние per-phone MFA durations, а выдаёт локальный residual/gate для эффективной длительности и acoustic conditioning. При нулевом gate поведение совпадает со старой моделью; при наличии MFA-контроля адаптер постепенно корректирует только нужные фонемы. CTC/duration loss сохраняются как стабилизатор произношения.

### 2026-09-11 — запущен полный MFA-duration adapter эксперимент

Для проверки гипотезы подготовлена отдельная ветка `transformer_aligned_phoneme_frankenstein_ru_mfa_duration_10k`. Полный русский корпус из `manifests/phoneme_v2` — `25 083` записей — передан MFA; рабочая папка использует hardlink на исходные WAV, поэтому исходный датасет не копируется и не изменяется. После выравнивания из CSV будет собран компактный кеш `cache/mfa_duration_ru_v1/<speaker>/<id>.npz` только с integer per-phone `duration_frames`, включая `<wb>` и согласованный с текущей фонемной последовательностью порядок. Полные промежуточные CSV сохраняются до проверки кеша.

В модель добавлен нулево-инициализированный `DurationConditioner`: он локально добавляет в acoustic hidden информацию о log-duration и относительном per-phone отклонении. Старый `duration_head`, CTC forced-alignment и существующие flow/x0/speaker/prosody objectives сохранены. Конфиг и запуск: `configs/transformer_aligned_phoneme_frankenstein_ru_mfa_duration_10k.yaml`, `scripts/run_frankenstein_ru_mfa_duration_10k.ps1`. Fine-tune должен стартовать от `checkpoints/transformer_aligned_phoneme_frankenstein_ru_rankloss_10k_latest.pt` (`step=34500`) с новым optimizer state и писать только в новый checkpoint/log.

На время MFA поставлен watcher `scripts/continue_mfa_duration_training.ps1`: после успешного получения не менее `20 000` alignment CSV он автоматически соберёт компактный duration-кеш и запустит отдельное `10 000`-шаговое обучение. Старый checkpoint и исходные аудио не перезаписываются.

После первого запуска обнаружилась ошибка не в модели, а в конвертере: standalone punctuation groups считались MFA-словами. Конвертер исправлен, кеш пересобран по уже готовым alignment CSV. Итоговый кеш `cache/mfa_duration_ru_v1` содержит `21 749/25 083` валидных записей; `3 334` записи исключены из-за оставшихся расхождений word segmentation, исходные CSV сохранены для аудита.

### 2026-09-11 — MFA-duration fine-tune фактически запущен

После smoke-проверки полного DataLoader на `frame_budget=5000` без OOM запущено реальное отдельное обучение `transformer_aligned_phoneme_frankenstein_ru_mfa_duration_10k` от rank-loss checkpoint `step=34500`. На старте совместимо загружено `320/326` tensor entries; шесть новых tensor entries относятся к нулево-инициализированному DurationConditioner. На шаге `150/10000` процесс стабилен, checkpoint пишется в новый файл, старый checkpoint не затрагивается. Первичная скорость около `1.5 с/шаг`, грубая ETA полного run — `4–5 часов` с учётом validation-точек.

Во время run свободное место на C: упало до `1.26 ГБ`, поэтому удалены только производные smoke/overfit-checkpoint’ы ранних 20-репличных тестов (`_smoke_textloss_from_123k`, старые `overfit_20`), суммарно около `3.55 ГБ`. Текущий MFA-duration checkpoint, rank-loss база, full-200k база, prosody-модели и логи не удалялись. После очистки свободно около `4.81 ГБ`.

Обучение остановлено пользователем после достижения целевого `step=10000/10000`; финальный checkpoint валиден и сохранён в `checkpoints/transformer_aligned_phoneme_frankenstein_ru_mfa_duration_10k_latest.pt`. Validation на `step=10000`: `loss=0.66391`, `mel_l1=0.32203`, `duration=0.10025`, `ctc=0.96657`, `prosody_loss=0.07709`, `ref_dependency=1.01083`. В процессе был исправлен прямой `torch.save` на запись во временный файл с атомарной заменой и fallback legacy serialization; повреждённый промежуточный файл удалён.

Для слухового сравнения созданы две одинаковые inference-сетки по 8 WAV: новая модель — `samples/mfa_duration_adapter_10000_grid/`, контрольная rank-loss модель `step=34500` — `samples/mfa_duration_rankloss_34500_grid/`. Для Geralt и Yennefer проверены `auto_duration`, реальные `mfa_duration`, `mfa_first_stressed_4x`, `mfa_all_stressed_3x`; seed `42`, `32` flow-steps, одинаковые тексты/длины, рядом сохранены оригиналы и `metadata.json`. В тестовый скрипт добавлена явная передача duration control в `DurationConditioner`, чтобы новые варианты действительно проверяли обученный адаптер, а не только ручной `frame_text`.

После остановки обучения создан отдельный карикатурный duration-тест для новой модели `step=10000`: `samples/mfa_duration_adapter_10000_cartoon_grid/`. Варианты: `auto_duration`, растяжение первой ударной фонемы `6x`, всех ударных `4x` и общей длины `1.5x`; параметры зафиксированы в `metadata.json`, рядом лежат оригинальные Geralt/Yennefer. Для чистого сравнения создана такая же сетка на rank-loss checkpoint `step=34500`: `samples/mfa_duration_rankloss_34500_cartoon_grid/`.

По запросу добавлен специализированный тест «каждое О длительностью 0.5 с». Точный фонемный вариант `all_o_0_5s` применяется к символам `o/ˈo`; расширенный `all_o_like_0_5s` — к текущим фонетическим реализациям русского О (`o/ˈo`, `ʌ`, `ɵ` и варианты ударения). Сгенерированы новая модель и контроль: `samples/mfa_duration_adapter_10000_all_o_halfsec/` и `samples/mfa_duration_rankloss_34500_all_o_halfsec/`. У Geralt в контрольной фразе нет точного `o`, но есть четыре O-like реализации (`ˈɵ`, `ʌ`, `ʌ`, `ʌ`), поэтому для него доступен расширенный вариант; у Yennefer есть точный `o` и расширенный вариант.

Слуховая оценка пользователя: контрольная rank-loss модель `step=34500` справилась с карикатурным растяжением фонемы О на порядок лучше новой MFA-duration adapter модели. Это опровергает предварительный вывод о безусловном улучшении от adapter. Интерпретация: старая модель уже умела использовать изменённый `frame_text` для локального растяжения; новый `DurationConditioner`, добавленный поверх того же `frame_text`, внёс конфликтующее акустическое смещение и усилил шум/артефакты. MFA-adapter на 10k не считать улучшением качества; rank-loss checkpoint оставить текущим лучшим baseline для duration control.

### 2026-09-11 — контрольная фраза с растянутым «О» и шёпотом

На старом rank-loss checkpoint `step=34500` сгенерированы две версии текста «Всееееее ХОЛОДНЕЙ, уже близко»: обычная и low-energy/low-voicing аппроксимация шёпота. Результаты: `samples/rankloss_34500_cold_whisper_control/`. Текст подан в фонемизатор без ручной правки; повторяющиеся `е` сохранились в фонемной последовательности. Отдельного обученного режима whisper в текущей модели нет, поэтому второй вариант является только просодийной аппроксимацией.

### 2026-09-11 — корректный focused prosody control test

Предыдущий тест с текстом «Всееееее ХОЛОДНЕЙ, уже близко» признан некорректным для проверки управления просодией: повтор букв в тексте не является надёжным способом задать длительность фонемы, а регистр не задаёт акцент. Поэтому без переобучения повторён тест на старом rank-loss checkpoint `step=34500` с нормальным текстом «Всё холоднее, уже близко».

В `samples/mfa_duration_rankloss_34500_focus_prosody_control/` сохранены исходная реплика и три inference-варианта с одинаковым `seed=42` и `32` flow-steps: `normal`, `cold_emphasis_e_0_5s`, `cold_emphasis_e_0_5s_whisper_tail`. Для focused-вариантов изменялись только явные управляющие входы: длительность ударной фонемы `ˈe` в слове «холоднее» принудительно задана примерно `0.5 s`, на MFA-интервал слова добавлены усиленные F0/energy, а в третьем варианте после слова добавлен low-energy/low-voicing tail как приближение шёпота. Текст, speaker, seed и checkpoint не менялись; исходный WAV скопирован только для сравнения, исходные данные и модель не изменялись. Подробные параметры записаны в `metadata.json`.

Слуховая проверка выявила, что этот тест всё ещё нельзя считать валидным: `normal` потерял межсловную паузу, растяжение слышалось на «уже близко», а low-voicing tail пришёлся не на тот участок. Причина найдена в самом тесте: использовался текст с запятой вместо манифестных многоточий, искусственная neutral-кривая вместо русского prosody-кеша, а локальный MFA-интервал масштабировался по общей длительности, не по фактическим phone-диапазонам.

### 2026-09-11 — focused prosody control v2

Создан исправленный inference-only прогон `samples/mfa_duration_rankloss_34500_focus_prosody_control_v2/` на том же старом rank-loss checkpoint `step=34500`. Использован точный манифестный текст «Всё холоднее… Уже близко…», реальный `cache/prosody_frankenstein_ru_model_v1` для Geralt и MFA-derived per-phone durations с паузой около `0.68 s` между «холоднее» и «уже». Русская frame-level prosody warped piecewise по MFA-словам и gaps; фокусное усиление и растяжение `ˈe` привязаны к фактическому диапазону слова «холоднее», а low-voicing tail начинается с «уже».

Сгенерированы `geralt_source_prosody_mfa_duration.wav`, `geralt_source_prosody_cold_emphasis_e_0_5s.wav` и `geralt_source_prosody_cold_emphasis_e_0_5s_whisper_tail.wav`; рядом сохранены исходник и `metadata.json`. Старый ошибочный прогон сохранён отдельно для истории, checkpoint и исходные данные не изменялись.

Слуховая оценка v2: контрольная версия с исходной prosody и MFA-паузой близка к оригиналу, но голос немного роботический. В варианте с акцентом слышится вариант «всё холодней… уже близко», голос дрожит; ожидаемого чистого локального акцента на «холоднее» нет. В варианте с тихим хвостом после «всё» возникает резкая просадка громкости, которая постепенно нормализуется к концу «близко», то есть изменение распространяется шире заданного участка. Это подтверждает, что вход управления просодией используется, но текущий глобальный prosody-memory декодер плохо удерживает локальность и частично смешивает энергию/voicing с акустическим паттерном фразы.

Уточнение пользователя: целевая растяжка должна была применяться к гласному первого слова «всё», а не к `ˈe` в «холоднее». В используемой фонемизации «ё» представлена как ударная `ˈɵ`; предыдущий v2-файл поэтому не является проверкой запрошенного участка.

### 2026-09-11 — focused prosody control v3: растяжение «ё» в «всё»

Повторён тот же inference-only тест с параметрами `focus_word=всё`, `focus_symbol=ˈɵ`. Новый каталог: `samples/mfa_duration_rankloss_34500_focus_prosody_control_v3_ves/`. Текст, RU prosody-кеш, MFA word/gap warp, seed `42`, `32` flow-steps и checkpoint `step=34500` сохранены; ударная `ˈɵ` задана примерно `0.5 s`, а quiet-tail начинается с «уже». Предыдущие v1/v2 результаты не перезаписывались.

Слуховая оценка v3: в варианте с акцентом `ё` в «всё» растянута идеально, но акцент через F0/energy либо отсутствует, либо слишком слабый для уверенного восприятия. В варианте с quiet-tail шёпот проявляется очень сильно и прежде всего как падение громкости: эффект распространяется от «вс…», включая растянутую «ё», до конца, вместо локального тихого участка после «холоднее». Следовательно, текущий duration/frame-text control даёт точное локальное управление длительностью, а непрерывная prosody-подача через глобальный memory path не даёт надёжного локального контроля энергии/voicing и может влиять на всю фразу.

### 2026-09-11 — план следующей архитектурной итерации prosody control (не запускался)

Наблюдение v3 разделяет два механизма: duration/frame-text branch локален и работает, а continuous prosody branch добавляет последовательность в общий Transformer memory, где acoustic cross-attention выбирает информацию глобально. Поэтому следующий кандидат не должен просто получать новые веса loss в том же месте.

План: сохранить текущий rank-loss checkpoint `step=34500` как baseline; добавить отдельную нулево-инициализированную continuous local-prosody ветку, которая после выравнивания prosody на выходные mel-кадры подаёт покадровый residual/FiLM непосредственно в acoustic hidden рядом с `frame_text`. Global prosody memory на первом прогоне выключить или заморозить, чтобы проверить чисто локальную гипотезу. Duration head, CTC и базовое выученное распределение длительностей не заменять.

Короткий acceptance-run: 5k шагов fine-tune от baseline с тем же RU-кешем; затем одинаковый seed/flow steps и четыре контрольных условия — исходная prosody, растяжение `ˈɵ` в «всё», локальный F0/energy-акцент на «всё», low-voicing только с «уже». Принимаем ветку только если длительность остаётся локальной, энергия/F0 меняются преимущественно внутри заданного слова, а quiet-tail не просаживает начало фразы. Для количественного контроля записывать отдельно channel sensitivity и отношение эффекта внутри выбранного интервала к эффекту вне него; добавление новых loss без этой проверки не считать полезным.

### 2026-09-11 — запущен local continuous-prosody 5k acceptance run

Существующий `LocalProsodyAlignedTransformerFlowStageAModel` расширен с FAcodec-only до `prosody.input=continuous`; его local frame branch теперь принимает текущий 4-канальный RU prosody-кеш. Global prosody-memory в этой архитектуре не используется, а local projection нулево-инициализирована. Добавлена совместимость с `duration_control`, чтобы ranking-loss и aligned pipeline сохраняли прежний интерфейс.

Подготовлены `configs/transformer_aligned_phoneme_frankenstein_ru_local_prosody_5k.yaml` и `scripts/run_frankenstein_ru_local_prosody_5k.ps1`. Запуск идёт отдельным процессом от сохранённого `checkpoints/transformer_aligned_phoneme_frankenstein_ru_rankloss_10k_latest.pt` (`step=34500`), с полным `cache/prosody_frankenstein_ru_model_v1`, новым checkpoint `checkpoints/transformer_aligned_phoneme_frankenstein_ru_local_prosody_5k_latest.pt` и отдельным CSV-log. Smoke на `1` шаге прошёл на CUDA без OOM; совместимо загружено `320/322` tensor entries, два tensor entries относятся к новой local projection.

На контрольной точке `step=500` train/validation завершились без ошибки; validation `loss=0.70711`, `mel_l1=0.33191`, `ctc=1.05106`, `prosody_loss=0.08537`, `ref_dependency=0.43228`. Это промежуточные числа, не итог качества. Процесс продолжает работать до `5000`; baseline и исходные данные не изменялись.

### 2026-09-11 — форматы аудио текущего пайплайна

Проверен представитель RU/EN/PL записи `geralt/158023`: исходные `ru.wav`, `eng.wav`, `pl.wav` — WAV PCM16, mono, `48 kHz`, без сжатия; эффективный PCM bitrate около `768 kb/s`. `load_mono` переводит waveform в float32 в памяти, сводит в mono и ресемплирует на `24 kHz`. Тренировочный `cache/audio_24k_mono` хранится как WAV PCM16 mono/24 kHz, около `384 kb/s`.

В training model напрямую получает не waveform и не bitrate, а log-magnitude mel: `n_fft=1024`, `hop_length=256`, `100` mel bins, natural log; на диске `cache/mel_v1` хранится как `float16 [frames, 100]`, при загрузке переводится в float32. Flow-модель работает с нормализованным log-mel и после sampling возвращает log-mel для Vocos. Vocos выдаёт float32 waveform в памяти, а `save_mono` сохраняет результат как PCM16 mono/24 kHz WAV — около `384 kb/s`.

Фактический размер на диске: основной `C:/witchervoicesdataset/dataset_staging` — `76.33 GB` (`71.11 GiB`, `448 863` файлов); весь `C:/witchervoicesdataset/witcher_tts/cache` — `10.77 GB` (`10.03 GiB`, `197 916` файлов). Весь кеш раздут прежде всего старым `cache/mfa_full_ru_work` — `8.69 GB`; актуальные для текущего local-prosody run `cache/mel_v1` — `1.69 GB` и `cache/prosody_frankenstein_ru_model_v1` — `0.25 GB`, вместе около `1.94 GB`. Свободно на C: около `3.56 GiB`.

### 2026-09-11 — разбор двух возможных оптимизаций хранения

Проверено, где теряется precision и нужен ли 24 kHz WAV-посредник. `build_mel_cache.py` уже читает исходный 48 kHz WAV, через `load_mono` ресемплирует его в float32 непосредственно в памяти и сразу строит mel; `cache/audio_24k_mono` в текущем training path не используется. Поэтому исходные 48 kHz записи не нужно конвертировать или перезаписывать: это не ускоряет обучение и нарушает сохранность исходников. Существующий audio-cache можно перестать создавать/удалить отдельным безопасным cleanup; сейчас он содержит только около `0.7 MB`.

`cache/mel_v1` действительно сохраняет рассчитанный float32 mel как `float16`, затем training загружает его обратно как float32. Полный float32 mel-cache потребует пересчёта из исходных WAV и будет примерно `3.38 GB` вместо `1.69 GB`; простое расширение старого float16-cache обратно в float32 качество не восстановит. Это можно проверить отдельным A/B после текущего run, но запускать такой пересчёт при свободных `3.56 GiB` рискованно из-за атомарной записи больших checkpoints. Отключение CUDA AMP — отдельный вопрос: оно не относится к формату cache и, вероятнее всего, только замедлит обучение/увеличит VRAM.
### 2026-09-11 — освобождение места и отдельный float32 mel-cache

После проверки ссылок удалён только неиспользуемый промежуточный каталог `cache/mfa_full_ru_work` размером `8 687 111 691` байт (`8.69 GB`). В рабочем коде он встречается лишь как default-путь старого MFA-скрипта; текущие модели используют готовые `mfa_full_ru_aligned`/`mfa_duration_ru_v1`, а активный cache-generation path его не читает. Исходный `dataset_staging`, актуальные alignment-кеши, чекпоинты и `mel_v1` не затронуты. Каталог удалён окончательно, восстановление возможно только повторным запуском MFA.

В `scripts/build_mel_cache.py` добавлены необязательные `--out-root` и `--dtype {float16,float32}` с сохранением прежних default-настроек. Запущена отдельная полная сборка `cache/mel_v2_float32` из исходных 48 kHz WAV с `dtype=float32`, поэтому `mel_v1` остаётся неизменным и текущие результаты воспроизводимы. Запуск detached: `logs/build_mel_v2_float32.log`; PID оболочки `924`, Python-процесса `11804`. Ожидаемый размер нового кеша около `3.38 GB`. Промежуточный `cache/audio_24k_mono` не включается: текущий builder его не использует.
Поскольку `cache/audio_24k_mono` не участвует ни в построении mel, ни в обучении, он также удалён как неиспользуемый промежуточный кеш (`694 826` байт). 48 kHz/24-bit или PCM-исходники не изменялись.
### 2026-09-11 — завершён local continuous-prosody 5k acceptance run

Проверочный run дошёл до `step=5000/5000` и штатно сохранил `checkpoints/transformer_aligned_phoneme_frankenstein_ru_local_prosody_5k_latest.pt`; это не остановка из-за ошибки. Финальная validation: `loss=0.63341`, `mel_l1=0.30964`, `duration=0.10790`, `ctc=1.04477`, `prosody_loss=0.05257`, `ref_dependency=0.43023`. Процесс завершён до начала перестройки mel-кеша; checkpoint и старый baseline не изменялись.

### 2026-09-11 — float32 mel-cache собран

Полный `cache/mel_v2_float32` собран из `25 083` записей (`25 083` `.npy`) за примерно `2 минуты`; итоговый размер `3 380 811 824` байта (`3.38 GB`). Проверка representative clip: `float32`, форма `[434, 100]`, форма совпадает с `mel_v1`; разница с сохранённым `float16` после обратного приведения — это только ранее потерянное квантование (на проверенном клипе max abs около `0.0039`). Новый кеш не подключён к старым конфигам автоматически: для A/B нужно явно указать `audio.mel_cache_subdir` на `cache/mel_v2_float32`. Свободно на C: около `22.1 GB`.
### 2026-09-11 — inference-тест готовой local-prosody модели

Скрипт `scripts/generate_cartoon_prosody_grid.py` расширен выбором класса по `model.architecture`, чтобы корректно тестировать `transformer_flow_aligned_local_prosody`. На checkpoint `step=5000` сгенерированы 8 вариантов на CUDA: `neutral`, `cartoon_bounce`, `cartoon_question`, `cartoon_staccato` для Geralt («Всё холоднее… Уже близко…») и Yennefer («И почему я не удивлена? Геральт, Геральт, что ты надел?»). В каждой паре seed `42`, `32` flow-steps и длина фиксированы; рядом сохранены оригиналы и `metadata.json`.

Результаты: `samples/local_prosody_5k_cartoon_prosody/`. Это inference-only проверка; модель, исходники и кеши не изменялись.
### 2026-09-11 — слуховая оценка local-prosody cartoon-теста

Пользователь прослушал `samples/local_prosody_5k_cartoon_prosody/`. Порядок вариантов: `1=neutral`, `2=cartoon_bounce`, `3=cartoon_question`, `4=cartoon_staccato`.

Geralt: `neutral` местами даёт хрип на вдохе; `cartoon_bounce` звучит так, будто голос зажат, и в целом грубее/злее; `cartoon_question` заметно меняет эмоциональную окраску; `cartoon_staccato` содержит резкие сильные провалы громкости. Yennefer: `neutral` в целом нормальный; `cartoon_bounce` звучит умоляюще/плаксиво; `cartoon_question` воспринимается как восторженно-радостный; `cartoon_staccato` также даёт резкие провалы громкости. В обоих случаях вариант 4 не является шёпотом.

Вывод: local continuous-prosody ветка действительно использует управляющий вход — одна и та же синтетическая кривая даёт разные, но систематические изменения у двух speaker. Однако каналы не разделены акустически: `F0/energy` меняют и качество фонации/эмоциональную окраску, а `voicing/pause` реализуются как скачки уровня, а не как чистая локальная пауза. Это подтверждает наличие просодийного контроля, но также показывает необходимость следующего теста с умеренными диапазонами и раздельной проверкой каналов; результат float32-кеша к этому тесту отношения не имеет.
### 2026-09-11 — следующий шаг после local-prosody 5k

Local continuous-prosody checkpoint `step=5000` признан достаточным для проверки гипотезы, поэтому продолжать его вслепую ещё на десятки тысяч шагов нецелесообразно. Синтетический cartoon-тест показал реальное влияние prosody-входа, но экстремальные и некоррелированные каналы вызывают speaker-зависимую фонацию, эмоциональную окраску и резкие провалы громкости.

Следующий эксперимент: inference-only channel-isolation grid на той же фразе и seed — умеренные независимые изменения F0, energy, voicing/pause и их комбинации — с контролем против старого rank-loss checkpoint. Цель — измерить рабочий диапазон каждого канала и отделить полезное управление от патологической реакции. Только если этот grid даст чистые локальные эффекты, готовить v2-контроллер с отдельными нулево-инициализированными channel branches/FiLM и realistic perturbation/dropout. float32 mel-cache использовать в следующем A/B, но не считать его главным способом улучшения.
### 2026-09-11 — инвентаризация кандидатов на удаление checkpoint-весов

Удаление пока не выполнялось. Для текущей ветки prosody обязательно оставить: `transformer_aligned_phoneme_full_200k_latest.pt` как чистый aligned init, `transformer_aligned_phoneme_frankenstein_ru_rankloss_10k_latest.pt` как проверенный baseline/родитель и `transformer_aligned_phoneme_frankenstein_ru_local_prosody_5k_latest.pt` как текущую модель. `transformer_aligned_phoneme_en_refer_20k_latest.pt` временно оставить до закрытия вопроса с иностранным референсом; `speaker_classifier.pt` оставить для диагностики.

Кандидаты: smoke и ранние прототипы (`_tmp_local_prosody_smoke.pt`, `stage_a_latest.pt`, `stage_a_vocos_latest.pt`, `stage_b_latest.pt`, оба `transformer_aligned_phoneme_from_raw_4k_*`) — около `1.70 GB`; старые/непродолжаемые prosody-ветки (`transformer_aligned_phoneme_full_100k_native_latest.pt`, `transformer_aligned_phoneme_ru_native_prosody_{20k,50k}_latest.pt`, `transformer_aligned_phoneme_ru_native_prosody_local_9k_latest.pt`, `transformer_aligned_phoneme_frankenstein_ru_mfa_duration_10k_latest.pt`, `transformer_aligned_phoneme_frankenstein_ru_prosodyloss_30k_latest.pt`) — около `3.07 GB`; старые grapheme/text-loss веса (`transformer_full_*` и `transformer_full_textloss_*`) — около `4.72 GB`. Опционально можно удалить обе старые FAcodec-prosody-translator модели `prosody_translator_full_{30k,8m_30k}.pt` — ещё `0.14 GB`, если ветка discrete FAcodec окончательно закрыта.

Все кандидаты можно восстановить только повторным обучением. Их логи, отчёты и уже созданные WAV не требуют хранения самих весов для статьи. Потенциальное освобождение всего списка — около `9.63 GB`; до отдельного подтверждения файлы не удалять.
### 2026-09-11 — удалены checkpoint-кандидаты и запущен channel-isolation тест

По прямому запросу удалены ровно 22 checkpoint-файла из групп 1–3 инвентаризации: smoke/ранние прототипы, закрытые просодийные ветки и старые grapheme/text-loss модели. Освобождено `10 189 835 656` байт (`10.19 GB`). Оставлены `full_200k`, `rankloss_10k`, `local_prosody_5k`, `en_refer_20k`, `speaker_classifier` и обе старые FAcodec prosody-translator модели из опционального пункта 4. Логи, отчёты и WAV-семплы сохранены; удалённые веса восстановимы только повторным обучением.

Для следующего решения сгенерирован inference-only channel-isolation grid на CUDA, `seed=42`, `32` flow-steps, одинаковые тексты/длины и две реплики Geralt/Yennefer. Варианты: `neutral`, `f0_only` (только умеренная высотная траектория), `energy_only` (только энергия), `pause_only` (только voicing/pause в локальном интервале), `moderate_combo` (умеренная комбинация). Одинаковый тест проведён для `local_prosody_5k` и `rankloss_34500`; в каждой папке по 10 синтетических WAV и оригиналы:

- `samples/channel_isolation_local_prosody_5k/`
- `samples/channel_isolation_rankloss_34500/`

Модель, исходный датасет и актуальные кеши не изменялись.
### 2026-09-11 — слуховая оценка умеренного channel-isolation grid

Пользователь прослушал `samples/channel_isolation_local_prosody_5k/` и `samples/channel_isolation_rankloss_34500/`. В обоих вариантах `pause_only` у Geralt и Yennefer даёт хрип вместо чистой паузы. У Yennefer local-модель заметно энергичнее на `energy_only`, чем rankloss; `f0_only` в local воспринимается скорее как немного менее громкая/энергичная подача, а не как уверенное изменение высоты. В целом различия между условиями и моделями микроскопические и плохо пригодны для надёжной слуховой интерпретации.

Уточнён вывод по local-prosody: экстремальные cartoon-кривые вызывают сильную, но патологически entangled реакцию; умеренные независимые кривые почти не слышны. Значит, текущая ветка не даёт одновременно силы и чистоты управления. Продолжать её тем же objective без изменения способа conditioning нецелесообразно; следующим шагом нужен контролируемый channel-gated/FiLM вариант либо возврат к базовой модели для отдельной проверки только duration/паузы.
### 2026-09-11 — уточнение результата pause-only и цели prosody translator

Уточнение пользователя: в `pause_only` локальная пауза присутствует и хрип не накладывается на соседнюю речь; модель заполняет сам интервал тишины хрипом/дыхательным шумом. Поэтому pause-control нельзя считать неработающим: временная граница и отделение от речи уже управляются, но акустическое содержимое паузного участка загрязнено.

Стратегическая цель не меняется: нужен переводчик иностранной просодии в русский просодийный control-space. Текущий local-prosody generator — это принимающая сторона, которую сначала нужно довести до чистой реализации control-сигналов. Следующая receiver-итерация должна сохранить уже рабочие MFA/duration-паузы, добавить отдельное подавление энергии/voicing внутри пауз и развести F0/energy по независимым gated/FiLM-веткам. После этого EN/PL prosody translator обучается на парных EN/PL→RU данных и подаёт результат в этот receiver; translator не отменяется и не откладывается как самостоятельная цель.
### 2026-09-11 — запущен receiver v2: gated local prosody + pause cleanup

Для ремонта принимающей стороны создана отдельная архитектура `transformer_flow_aligned_gated_local_prosody`. Она стартует от чистого `rankloss_10k` baseline и не использует уже entangled `local_prosody_5k`: четыре continuous-канала получают независимые bounded FiLM-ветки с нулевой инициализацией, а пауза дополнительно проходит через мягкий low-voicing silence gate. Базовые text/speaker/duration/CTC/flow-механизмы сохранены.

В `train_stage_a.py` добавлен `pause_silence_loss`: на кадрах с `pause_score>=0.5` он отдельно сопоставляет энергию predicted `x0` с целевым low-energy mel, чтобы паузный интервал не заполнялся хрипом. Все новые метрики пишутся в CSV. Подготовлены конфиг `configs/transformer_aligned_phoneme_frankenstein_ru_gated_local_prosody_5k_float32.yaml` и runner `scripts/run_frankenstein_ru_gated_local_prosody_5k_float32.ps1`; run использует полный `cache/mel_v2_float32`, 8 DataLoader workers, `5000` шагов и отдельный checkpoint `checkpoints/transformer_aligned_phoneme_frankenstein_ru_gated_local_prosody_5k_float32_latest.pt`.

Smoke на полном DataLoader/CUDA прошёл: загружено `320/346` базовых tensor entries, новый `pause_silence` loss считается. Полный run запущен detached; на первых `50` шагах ошибок/OOM нет. Все старые модели и исходные данные сохранены.
### 2026-09-11 — первый gated receiver run остановлен по CUDA OOM

Repair-ветка `transformer_flow_aligned_gated_local_prosody` была запущена от `rankloss_10k` на float32 mel-cache. Smoke прошёл, но полный run остановился на `step=750` из-за CUDA OOM при очередном пике длины batch: PyTorch пытался выделить ещё `124 MiB` при уже занятом лимите 8 GiB. Это не ошибка новой метрики или несовместимость checkpoint. Остановленный артефакт сохранён отдельно как `checkpoints/transformer_aligned_phoneme_frankenstein_ru_gated_local_prosody_5k_float32_oom_step500.pt`, CSV переименован в `..._oom.csv`.

Для retry оставлены та же архитектура и objective, но `frame_budget` снижен с `5000` до `3500`, DataLoader возвращён к `4` workers, а runner получил `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Retry пишет в отдельные `..._retry_latest.pt` и `..._retry.csv`; старый local checkpoint и rankloss baseline не изменяются.
Retry gated receiver реально стартовал на `2026-09-11 20:27`; на `step=150` жив, loss считается, новый checkpoint ещё не финальный. Первичная скорость после прогрева около `0.6 с/шаг`; предварительная ETA `21:15–21:25`. Предупреждение о неподдержанном `expandable_segments` не является ошибкой; основной safety fix — сниженный frame budget.
### 2026-09-11 — предварительная оценка масштабирования 46M → 150M/300M

Текущая aligned-модель фактически содержит `45 967 485` параметров (то есть около `46M`, а не ровно 44M). Полный phoneme-v2 корпус содержит около `49.9` часов RU-аудио, из них около `22.45` часов train. По последним контрольным логам train mel-L1 около `0.307`, validation mel-L1 около `0.336`: явного сильного train/validation overfit не видно, но это не доказывает, что основной bottleneck — ёмкость.

При сохранении архитектурных пропорций переход к `150M` требует примерно `3.26x` параметров и ширины около `1.81x`; к `300M` — `6.53x` параметров и ширины около `2.56x`. Для текущего 8 GB GPU это означает ориентировочно `1.7 GB` и `3.4 GB` на checkpoint с optimizer state против текущих `~0.53 GB`, а память активаций остаётся главным ограничением. Training 150M потребует заметно меньшего frame budget/activation checkpointing, 300M — уже практически обязательного offload или другого GPU.

Предварительный вывод: 150M может дать умеренное улучшение акустической детализации/обобщения при достаточном compute, но не обязано исправить смешение prosody-каналов; 300M без увеличения данных и изменения objective с высокой вероятностью даст diminishing returns или overfit. Точный прирост качества по параметрам заранее не вычисляется; его можно оценить только через контролируемый scaling pilot, но текущие train/validation и memory diagnostics уже позволяют не ожидать линейного улучшения.
### 2026-09-11 — оценка расширения корпуса до всех спикеров >10 минут

По готовому анализу `C:\witchervoicesdataset\dataset_analysis\speaker_summary.csv` текущий phoneme-v2 manifest использует 10 спикеров и `24.95` ч русской речи целиком (`22.45` ч train). В полном staging-корпусе порог строго `RU > 10 минут` проходят `65` спикеров (`40.65` ч RU). Значит, добавление всех проходящих спикеров даст ещё `55` speaker-slug, `10 880` клипов и `15.70` ч RU — примерно `+63%` к объёму текущей десятки. Для тех же записей доступно около `15.11` ч EN и `14.40` ч PL.

В этих 55 дополнительных группах есть `unknown` и `mirror` суммарно около `1.07` ч RU. `mirror` по уточнению пользователя, вероятно, является Гюнтером о’Диммом, поэтому это не безымянный мусор и автоматически исключать его нельзя; отдельно проверить нужно только качество атрибуции. Если временно убрать лишь `unknown`, остаются `54` дополнительные группы и около `15.12` ч RU.

Численная оценка не является измеренным приростом: прямой A/B ещё не проводился. При speaker-balanced sampling и неизменном качестве данных расширение, вероятно, даст поверх ожидаемых от 46M→150M примерно `+3–8%` относительного улучшения по оставшейся акустической ошибке/обобщению, то есть ориентировочно `+2–6 процентных пунктов` к условным `10%` эффекта одного масштабирования. Для разборчивости новых текстов эффект вероятнее заметен, чем для сходства именно с Geralt/Yennefer: их собственные часы не увеличиваются. При наивном смешивании по клипам Geralt будет подавлять остальных, а target-качество может получить `0%` или ухудшиться; добавление голосов само по себе не чинит паузный хрип и entangled prosody.

Итоговая рабочая гипотеза: `150M + 53–55` дополнительных спикеров может дать порядка `12–18%` относительного снижения остаточной акустической ошибки против текущей 46M/10-speaker базы, но это не MOS и не гарантия слышимого прироста. Проверять нужно парой одинаковых прогонов с capped/speaker-balanced sampler; иначе вклад размера и вклад корпуса неразделимы.
### 2026-09-11 — A/B gated receiver против rankloss baseline

После завершения gated retry проведён inference-only A/B на тех же условиях, что и прежний channel-isolation тест: `seed=42`, `32` flow steps, одинаковые Geralt/Yennefer, текст, target duration и пять вариантов prosody (`neutral`, `f0_only`, `energy_only`, `pause_only`, `moderate_combo`). Получено 10 WAV от `gated_local_prosody` (`step=5000`) и 10 WAV от родительского `rankloss` (`step=34500`), плюс контрольные оригиналы. Файлы собраны в `samples/ab_gated_local_prosody_5k_float32/` и `samples/ab_rankloss_34500/`; индекс с прямыми ссылками — `reports/gated_vs_rankloss_ab.md`.

Исходники, checkpoint и обучение не изменялись. Цель A/B — на слух проверить, стала ли новая развязка каналов чище и сильнее, особенно в `pause_only` и `moderate_combo`; численные loss-метрики сами по себе этого не доказывают.
### 2026-09-11 — gated local prosody retry завершён на 5k

Безопасный retry `transformer_aligned_phoneme_frankenstein_ru_gated_local_prosody_5k_float32_retry` штатно достиг `step=5000/5000` и записал validation. CUDA OOM не повторился после снижения `frame_budget` до `3500`. Финальная validation: total loss `0.63159`, mel-L1 `0.31791`, duration `0.09561`, speaker `0.0000156`, x0 `0.01963`, CTC `0.92897`, общий prosody-loss `0.04717`; его компоненты F0 `0.02204`, voicing `0.55622`, energy `0.04245`, pause `0.12795`, delta `0.00475`; `pause_silence=0.000689`, `ref_dependency=0.33836`.

Артефакты: `checkpoints/transformer_aligned_phoneme_frankenstein_ru_gated_local_prosody_5k_float32_retry_latest.pt` и соответствующий CSV в `logs/`. Прогон завершён, но сами метрики не заменяют слуховой A/B: они подтверждают стабильное обучение и прямое влияние нового loss, а не автоматически чистое звучание.
### 2026-09-11 — слуховая оценка A/B gated receiver и системный «zombie» артефакт

Пользователь прослушал `samples/ab_gated_local_prosody_5k_float32/`. У Geralt `pause_only` лучше всего подавляет хрип в середине паузы; `neutral` подавляет его слабее; `moderate_combo` даёт отчётливый акцент на «всё холоднее»; `f0_only` — небольшой акцент на «уже близко»; `energy_only` меняет энергию, но в паузе вместо хрипа появляется «ээээ». У Yennefer `pause_only` даёт нормальную паузу без хрипа. Это подтверждает, что gated receiver уже управляет паузой/энергией/акцентом, но каналы ещё связаны с фонацией.

Отдельно зафиксировано системное наблюдение: все актёры и фразы звучат как «зомби» — речь приблизительно есть, но темп, артикуляция и естественная фонация деградированы. Проблема присутствовала ещё до prosody-ветки, поэтому это не побочный эффект gated-модуля и не проблема конкретного спикера. Предыдущие `~25%` — условная оценка снижения остаточной акустической ошибки при удачном масштабировании и расширении данных, а не обещание `25%` субъективной естественности; этот прогноз нельзя переносить на «зомби»-эффект без отдельной диагностики mel→vocoder.

Перед переходом к 150M или добавлением W2 нужен oracle-тест: декодировать исходный target-mel тем же Vocos и сравнить его с синтезом. Если oracle-mel звучит нормально, bottleneck в acoustic flow/обучении; если тоже «зомби», сначала исправлять mel/vocoder frontend mismatch. Добавление спикеров само по себе системную фонацию не исправит.
### 2026-09-11 — выполнен Vocos oracle-тест

Выполнена проверка на тех же контрольных репликах Geralt `0x00026947` и Yennefer `0x000286fa`: в frozen `charactr/vocos-mel-24khz` подан настоящий target из `cache/mel_v2_float32`, без Transformer, текста, speaker conditioning или checkpoint. Для контроля исходник пересохранён в 24 kHz тем же audio loader. Vocos oracle длительности практически совпали с оригиналом: Geralt `3.040` против `3.050` с, Yennefer `6.784` против `6.787` с.

Семплы: `samples/vocos_oracle_reconstruction_geralt_yennefer/`; отчёт со ссылками: `reports/vocos_oracle_reconstruction.md`. Следующий диагностический вывод зависит от слуха: если oracle чистый, системный «зомби»-эффект рождается в acoustic Transformer/его обучении; если oracle также искажён, сначала чинится frontend mel↔Vocos.
### 2026-09-11 — oracle признан на слух идентичным оригиналу

Пользователь сравнил обе пары и сообщил, что Vocos oracle и оригиналы на слух идентичны. Frontend `LogMel → Vocos`, sample rate, mel-конфигурация и декодер тем самым исключены как источник системного «зомби»-эффекта. Дефект появляется в acoustic Transformer: модель генерирует недостаточно естественный mel при восстановлении речи из текста/шума, хотя тот же Vocos корректно воспроизводит настоящий mel.

Следствие для планирования: рост до 150M и расширение корпуса могут улучшить acoustic mapping, но прежнюю оценку `~25%` нельзя считать обещанием субъективной естественности. Нужны отдельные проверки flow sampling (12/32/64 steps) и teacher-forced/noisy-mel prediction, чтобы отделить ошибку интеграции от ошибки самого acoustic objective; добавление спикеров или Witcher 2 без этого не гарантирует устранение «зомби».
Уточнение: flow sampling `12/32/64` уже проверялся ранее на full Transformer; увеличение числа шагов проблему не исправило. Поэтому повторять этот sweep как следующий шаг не нужно. Остаётся teacher-forced/noisy-mel диагностика и проверка самого acoustic objective.
### 2026-09-11 — teacher-forced/noisy-mel диагностика завершена

На тех же контрольных Geralt/Yennefer проверены `x_t=(1-t)*target_mel+t*fixed_noise` для `t=0.15/0.50/0.85`; использованы neutral prosody и модельный CTC-derived frame/text alignment. Для gated step 5000 средние значения `x0-L1` по зонам: `0.069/0.167/0.308`; для rankloss step 34500: `0.073/0.169/0.459`. В noise-dominant зоне gated лучше примерно на `33%` по flow-L1; для Yennefer `x0-L1=0.298` против `0.554`, для Geralt `0.319` против `0.364`.

Это показывает, что acoustic path умеет корректировать почти готовый target-mel, но заметно теряет содержание при восстановлении из шума. Gated retry улучшает этот участок относительно родителя, но абсолютная ошибка всё ещё достаточно велика, чтобы давать системный «зомби»-эффект. Сохранены декодированные `x0` WAV в `reports/teacher_forced_flow_diagnostic/x0_audio/`, численные таблицы и отчёт `reports/teacher_forced_flow_diagnostic.md`. Vocos и sampling steps уже исключены; основной кандидат теперь acoustic prior/objective/данные.
### 2026-09-11 — предварительная оценка добавления корпуса Witcher 2

В локальном проекте отдельная база Witcher 2 и её manifest пока не обнаружены, поэтому оценка условная и зависит от числа действительно параллельных EN/RU/PL часов, чистоты атрибуции и совпадения актёров. Для текущей гипотезы `150M + расширенный Witcher 3 (>10 минут)` добавление качественного W2 не даёт линейного бонуса: ориентировочно ещё `+1–3` процентных пункта при ~10 ч W2, `+3–6` п.п. при ~25 ч и `+5–9` п.п. при ~50 ч. Это оценка относительного снижения остаточной акустической ошибки, не MOS.

Если W2 содержит тех же целевых русских актёров и хорошо выровненные параллельные реплики, верхняя граница вероятнее; если это другие актёры/мастеринг или только непарная RU-речь, эффект в основном ограничится фонетическим и акустическим обобщением и может быть `0–2` п.п. для сходства Geralt/Yennefer. С учётом прежней оценки `12–18%` для `150M + W3>10m` рабочий диапазон с хорошим W2 — примерно `15–25%` против текущей 46M/10-speaker базы, но только при speaker/domain-balanced sampling; эффекты не складываются арифметически.
### 2026-09-11 — следующий шаг: scaling acoustic prior + расширенный Witcher 3

После oracle и teacher-forced/noisy-mel тестов основной рабочий bottleneck — acoustic prior Transformer при восстановлении mel из noise. Следующий осмысленный пилот: aligned/gated receiver увеличенного размера около `150M` на расширенном Witcher 3-корпусе (`65` speaker-групп, `40.65` ч RU целиком вместо `24.95` ч; при исключении только `unknown` — `64` группы и около `40.07` ч RU). Сохраняются тот же mel/frontend, Vocos, текстовая фонемизация, оценочные WAV и noisy-mel контроль.

Это проверяет гипотезы ёмкости и разнообразия acoustic prior. На текущей 8 GB GPU понадобятся уменьшенный frame budget, вероятно activation checkpointing/gradient accumulation и более медленный, но безопасный режим. `300M` пока не является следующим практичным шагом: риск OOM и стоимость выше, а причинно он ничего не объяснит.

Witcher 2 добавляется после базового пилота отдельным фактором. Иначе одновременно изменятся размер, число спикеров, домен и мастеринг, и статья не сможет показать, что именно убрало «зомби». Первый обязательный A/B: `46M/10 speakers` против `150M/65 speakers` на тех же контрольных фразах и noisy-mel зонах.
### 2026-09-11 — уточнён и запущен чистый 150M scaling-пилот

Постановка зафиксирована окончательно: включаются все speaker-группы с русской длительностью строго `>10 минут`, включая `mirror` (рабочая атрибуция — Гюнтер о’Димм), а `unknown` исключается целиком. Получен изолированный манифест на `64` группы и `35 661` параллельную запись: `32 066` train, `1 778` validation, `1 817` test. Существующие 10-speaker manifests, кеши и checkpoints не перезаписываются.

Подготовлена конфигурация `configs/transformer_aligned_gated_local_prosody_150m_64spk_100k.yaml`: та же aligned/gated local-prosody архитектура и objective, но ширины `text=640`, `prosody=384`, `acoustic=960`, `alignment=384`, `64` speaker embeddings; фактический размер — `151 420 309` параметров. Прогон намеренно начинается с нуля: старый 46M checkpoint несовместим по ширинам и числу speaker-классов, а частичная загрузка дала бы нечистое сравнение.

Для нового эксперимента собран отдельный float32 mel-cache `cache/mel_v3_float32_64spk` (`35 661/35 661` файлов) и строится отдельный RU Frankenstein prosody-cache `cache/prosody_frankenstein_ru_model_v2_64spk`. Добавлен параметр `--manifest-dir` в готовый mel-cache builder, чтобы исключить случайное чтение старого 10-speaker manifest. Runner `scripts/run_aligned_gated_local_prosody_150m_64spk_100k.ps1` использует `max-seconds=30` (максимум в manifest — `25.08` с), `frame_budget=1400`, mixed precision, 6 DataLoader workers, checkpoint/log отдельные; цель — `100 000` шагов. Сначала обязательны smoke и memory-check на CUDA, затем full detached run.

Уточнение к фактическому запуску: после smoke на `2200` кадрах без CUDA OOM (примерно `4.44/8.15 GB` VRAM) production `frame_budget` повышен до `3500`. Пилот на 2200 шагах/кадрах остановлен на `step=250` без потери checkpoint и сохранён отдельно в `logs/transformer_aligned_gated_local_prosody_150m_64spk_100k_budget2200_pilot.csv`. Он показал около `90%` GPU utilization; остановка была сделана только для более полной утилизации памяти.

В `2026-09-11 22:36` запущен чистый detached full run с `frame_budget=3500`, `mixed precision`, `6` workers и целью `100 000` шагов. На `step=150` процесс жив, GPU достигает `93%`, VRAM около `5.45/8.15 GB`; CUDA OOM не наблюдается. На старте сохранения ещё нет (первый checkpoint ожидается на `step=1000`). Лог: `logs/transformer_aligned_gated_local_prosody_150m_64spk_100k.csv`; runner и checkpoint остаются отдельными от старого 10-speaker корпуса.

Прогон с `frame_budget=3500` остановлен на `step=650` до первого сохранения: GPU доходил до `100%`, VRAM до `7.89/8.15 GB`, но private memory процесса выросла примерно до `15.2 GB`, из-за чего Windows раздул pagefile и свободное место падало до `3.4 GB`. Чекпоинт не существовал, поэтому сохранён только лог диагностики `..._budget3500_aborted_step650.csv`. Для продолжения выбран безопасный компромисс `frame_budget=2600`: он должен удержать хорошую загрузку GPU без опасного swap-pressure; целевой запуск остаётся тем же `150M/64 speakers/100k steps` и стартует с нуля.

В `2026-09-11 22:42` production run перезапущен с `frame_budget=2600`. На `step=200` private memory процесса около `8.36 GB`, свободно `16.25 GB`, GPU в замерах `~71–96%`, VRAM `~4.8 GB`; pagefile перестал быстро расти. Предварительная скорость `~2.3 step/s`, ETA по первым минутам `~12 ч` до 100k, с поправкой на validation/save реалистично закладывать `11–13 ч`. Это безопасный текущий режим; менять budget посреди чистого прогона не планируется.

После продолжения run достиг `step=2000`, но сохранение очередного checkpoint завершилось ошибкой Windows `No space left on device`: во время сериализации одновременно требовались старый checkpoint и временная копия при разросшемся pagefile. Валидный `transformer_aligned_gated_local_prosody_150m_64spk_100k_latest.pt` проверен чтением через PyTorch и содержит `step=1000`; незавершённый `.tmp` удалён как мусор, свободно около `24.8 GB`. Обучение остановлено до отдельного решения по освобождению места/режиму сохранений. Последний CSV сохраняет метрики до `step=2000` и validation на `step=2000`; полезный checkpoint для resume — `step=1000`.

По запросу удалены только WAV из `C:\witchervoicesdataset\dataset_staging`: из `192 368` файлов (`70.93 GB`) удалены `128 245` (`47.49 GB`, `66.67%`). Оставлены `64 123` WAV (`23.44 GB`), включая все `5 451` RU/EN/PL-файла текущего test-набора. `txt`, `meta.json`, manifests и все кеши не изменялись; test-проверка показала `0` отсутствующих контрольных аудио. После очистки свободно около `70.05 GB`.

В `2026-09-11 23:01` runner исправлен на явный `--resume` и перезапущен с валидного checkpoint `step=1000`, `frame_budget=2600`, цель `100 000`. На `step=1200` процесс жив, GPU около `94%` в пике, VRAM около `4.8/8.15 GB`, свободно около `47.4 GB`; лог возобновлённого прогона пишется в отдельный основной CSV. WAV в этом запуске не читаются: mel и prosody полностью берутся из кешей.

В `2026-09-12 00:07` для текущего 150M-прогона построен PNG-снимок лоссов: последний train-row — `step=9550`, последняя validation-точка — `step=9000`. Для сопоставления зафиксирована последняя validation-точка прежней aligned 46M-модели с просодией: локальный шаг `34500`, но её накопленная история — база `122000` + per-channel prosody-loss до `25000` + ranking-loss до `34500`, то есть около `156500` шагов суммарно. Предыдущая таблица была подписана недостаточно точно и сравнивала эту модель только по локальному номеру `34500`; это исправлено в интерпретации. Сравнение всё равно диагностическое, а не прямой рейтинг качества, поскольку отличаются корпус, число спикеров и стадия обучения.

В `2026-09-12 00:16` построены исправленные полноисторические сравнения в `reports/training_losses/compare_150m_vs_46m_full_2026-09-12/`: `main_losses_full_comparison.webp`, `prosody_losses_full_comparison.webp` и `reference_diagnostics_full_comparison.webp`. Старая модель отображена по кумулятивной оси до `156500`, текущая 150M — до фактически доступного `step=14750` на момент построения.

В `2026-09-12 00:21` PNG обновлены до текущего `step=15200`. На validation `step=15000` текущие значения: total `1.0433`, mel-L1 `0.4307`, CTC `2.4824`, duration `0.2415`, x0 `0.03618`. После `11k` mel-L1 почти стабилен около `0.43`, но CTC, duration и x0 ещё снижаются. Это делает сценарий высокого основного плато правдоподобным, но пока не подтверждает его окончательно.

В `2026-09-12 06:10` PNG обновлены в безоконном режиме до последнего train `step=18150` и validation `step=18000`. Текущая фактическая скорость от возобновления на `step=1000` — около `0.69` шага/с, что даёт грубую ETA `~33 часа` до `100000`; короткий 15-секундный срез без нового CSV-row нормален при такой скорости, поскольку логируется каждый 50-й шаг. При этом private memory Python-процесса около `21.2 GB`, свободной RAM около `1 GB`, поэтому ETA чувствительна к paging.

На validation `step=18000` CTC равен `2.417` против `3.497` на `step=2000` и `2.650` на `step=9000`: это замедление, похожее на плато, но не полная остановка. CTC остаётся главным индикатором риска для содержания/произношения; если к `30k` он останется выше примерно `2.0`, а mel-L1 выше `0.40`, joint-режим следует считать неэффективным и переходить к speech-first curriculum с последующим подключением prosody. До `30k` текущий запуск оставлен как диагностический.

Для слуховой проверки обучение остановлено на сохранённом checkpoint `step=18000` (несохранённые шаги не использованы). Сгенерирована inference-сетка `samples/current_150m_listen/`: Geralt и Yennefer, фраза из контрольного набора, условия `neutral`, `cartoon_bounce`, `cartoon_question`, `cartoon_staccato`, `32` flow-steps, seed `42`. Поскольку соответствующие исходные WAV были удалены при очистке, длительность взята из `cache/mel_v3_float32_64spk`; модель и кеш не изменялись. После прослушивания обучение можно возобновить через `--resume`.

Слуховой результат checkpoint `step=18000`: вместо речи слышен львиный рык/грубый шум, а по пользовательскому описанию голос между вариантами не меняется. Поэтому speaker/timbre conditioning нельзя считать подтверждённым: низкий speaker-loss сам по себе доказывает лишь работу внутреннего классификационного сигнала, но не слышимую идентичность голоса. Phoneme→acoustic/content path явно не выучен; карикатурная просодия лишь усиливает выход за пределы устойчивого acoustic manifold. Результат подтверждает риск по высокому CTC, но сам по себе не доказывает, что нормальная речь не появится после дальнейшего обучения.

По решению пользователя запущена отдельная speech-first ветка `transformer_aligned_speech_only_150m_64spk_30k`: новая модель обучается с нуля на том же 64-speaker manifest и float32 mel-cache, но без `--prosody-root`, без prosody reconstruction, pause-silence и reference-ranking. Continuous prosody/gated-модули оставлены в архитектуре и checkpoint, поэтому после `30k` можно совместимо подключить RU prosody-cache через `--init-checkpoint`, не смешивая эту ветку с загрязнённым joint-run. Smoke на 3 шагах прошёл; full run жив, первые `250` шагов записаны. Артефакты: `configs/transformer_aligned_speech_only_150m_64spk_30k.yaml`, `scripts/run_aligned_speech_only_150m_64spk_30k.ps1`, `checkpoints/transformer_aligned_speech_only_150m_64spk_30k_latest.pt`, `logs/transformer_aligned_speech_only_150m_64spk_30k.csv`.

Уточнение интерпретации: против финальной validation старой 46M prosody-модели (накопленно около `156.5k`) текущая 150M на `15k` имеет отношения примерно `1.45x` по total, `1.28x` по mel-L1, `2.38x` по CTC, `2.23x` по duration и `2.09x` по x0. При этом против старой plain 46M базы на сопоставимом раннем `15k` текущая также хуже (`mel-L1 0.431 vs 0.352`, `CTC 2.482 vs 1.435`). Поэтому график пока не доказывает «огромное бутылочное горлышко размерности»: 150M быстро подгоняет просодийные признаки, но основной acoustic path отстаёт. Вероятные смешанные причины — более трудный 64-speaker корпус, одновременное обучение всех objective с нуля, более низкие `LR/warmup` и меньший frame budget, а не только ёмкость модели. Нужна контрольная точка около `30k`, прежде чем объявлять plateau.

Рабочая гипотеза «два стула»: старая 46M сначала около `122k` шагов выучила устойчивое text/speaker→speech mapping, а затем просодийный fine-tune с маленькими нулевыми ветками почти не смог перестроить этот acoustic prior. Новая 150M учит speech и prosody одновременно, поэтому дешёвая auxiliary-просодия оптимизируется быстро, пока содержательная речь ещё нестабильна. Однако на validation `15k` просодийная часть не доминирует скалярный total: flow `0.4307`, weighted CTC `0.4965`, weighted x0/duration/speaker около `0.0424`, prosody `0.0308`, reference-ranking около `0.0405`, pause-silence `0.0025`. Значит, проблема вероятнее в конфликте градиентов/совместном представлении и curriculum, а не в том, что prosody-loss численно «съедает» обучение. Решающий A/B — два продолжения от одного 150M checkpoint: speech-only objective против текущего joint objective, с последующим одинаковым prosody-control тестом.

Дополнительный корпус не считается доказанным источником ухудшения. Он расширяет распределение: `64` speaker-группы, больше акустических условий и потенциально больше ошибок атрибуции/выравнивания; при `speaker_sampling_power=0.35` редкие группы получают относительно больше внимания. Поэтому средний validation-loss может быть выше и для выхода на прежнее качество требуется больше обновлений, но это не означает более низкий конечный потолок или потерю голоса Geralt/Yennefer. Нынешний A/B загрязнён одновременной сменой корпуса, размера, objective и режима старта; для изоляции вклада данных нужен хотя бы speech-only 150M на том же 64-speaker корпусе с теми же `LR/frame_budget`, а затем сравнение с joint-веткой.

В `2026-09-12 08:30` speech-first 150M-прогон завершился CUDA OOM во время `backward` около шага `8151`. Это не был OOM диска/файла подкачки: свободно около `58.7 GB` на C:, `6.64 GB` RAM и `21.69 GB` virtual memory; GPU после завершения свободен. Последний сохранённый checkpoint проверен по наличию и времени записи: `transformer_aligned_speech_only_150m_64spk_30k_latest.pt`, около `1.63 GB`, шаг `8100`; CSV содержит метрики до `8100`, validation до `8000`. Оригинальный runner и checkpoint не удалялись и не перезаписывались диагностикой.

Для безопасного продолжения создан отдельный `scripts/run_aligned_speech_only_150m_64spk_30k_resume_8100.ps1`: он возобновляет тот же checkpoint с восстановлением optimizer/scaler/scheduler, сохраняет цель `30000`, отключает prosody-loss как и исходная speech-first ветка и снижает только dynamic `frame_budget` с `2600` до `1800`; добавлена стандартная настройка `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Это изменение размера микробатча, а не архитектуры/данных/objective, поэтому возможность последующего prosody fine-tune сохраняется.

В `2026-09-12 08:35` recovery успешно прошёл аварийную область: после resume с `8100` достигнут `step=9100`, validation `step=9000` записана (`total loss≈1.022`), новый checkpoint сохранён около `08:35:29`, в recovery error-log новых OOM нет. Текущая скорость после загрузки — примерно `4.9` шага/с, грубая ETA до `30000` — около `1 ч 10–20 мин` с учётом validation/save. На этом этапе speech-only прогон продолжается в безопасном режиме `frame_budget=1800`.

В `2026-09-12 13:00` speech-first прогон штатно завершён на `30000/30000`: финальная validation записана и checkpoint сохранён. Validation: total `0.8961`, mel-L1 `0.3737`, duration `0.2207`, speaker `0.00034`, x0 `0.0360`, CTC `2.4117`. После recovery с `frame_budget=1800` новых CUDA OOM не было; процесс завершился с кодом успеха. Финальный файл: `checkpoints/transformer_aligned_speech_only_150m_64spk_30k_latest.pt` (около `1.63 GB`).

В `2026-09-12 13:07` построено PNG-сравнение двух 150M aligned моделей на общей оси шагов: joint + prosody (показана до последнего сохранённого шага `18000`) против speech-only (до `30000`). Основные лоссы собраны в `reports/training_losses/compare_150m_speech_only_vs_joint_2026-09-12/main_losses_speech_only_vs_joint_150m.png`; отдельная панель joint-просодии — в `joint_prosody_losses_150m.png`. Speech-only намеренно не имеет prosody-series: её prosody objective был отключён, но модули архитектуры оставлены для будущего fine-tune.

В `2026-09-12 13:10` на финальном speech-only checkpoint `step=30000` сгенерирована inference-сетка из `12` WAV: три русские реплики, исходно принадлежащие Geralt и Yennefer, каждая озвучена обоими speaker IDs. Использован готовый flow-generator, дополненный поддержкой gated aligned архитектуры и frame count из mel-cache при удалённых исходных WAV; prosody argument не подавался, чтобы не активировать необученную ветку. Результаты: `samples/speech_only_150m_30k_known_text_grid/`, `flow_steps=32`, `seed=42`, model/source не изменены.

Слуховая оценка `2026-09-12 13:12`: в output speech-only под доминирующим «инопланетным» рыком местами разбираются речь и даже идентичность speaker. Это указывает, что text/content и speaker path уже дают устойчивые признаки, но acoustic/flow output остаётся сильно вне нормального mel-распределения; низкий speaker-loss здесь не трактуется как доказательство чистого голоса. Главный следующий bottleneck — подавление рыка/стабилизация acoustic decoder, а не добавление просодийных loss.

В `2026-09-12 13:14` проверен вопрос о плато speech-only: жёсткого полного плато нет, но модель явно замедлилась. Validation от `15k` к `30k`: total `0.9282→0.8961`, mel-L1 `0.3823→0.3737`, CTC `2.5222→2.4117`, duration `0.2577→0.2207`. На последних точках `25k→30k` CTC практически стоит (`2.4088→2.4117`), mel колеблется вокруг `0.37–0.39`, total вокруг `0.88–0.91`. Поэтому это плато/насыщение речевого objective, но не доказательство полной бесполезности дальнейших шагов.

В `2026-09-12` проведён forensic A/B-аудит усиления артефактов 150M speech-only против старой 46M aligned модели. Основная причина не в Vocos, float32 или несовместимом inference: текущий checkpoint загружается с `0` missing/`0` unexpected, оба теста используют `24 kHz`, `32` flow steps, seed `42` и `286` frames; целевые mel старого float16 и нового float32-cache практически совпадают. Причины: новая модель стартовала с нуля и получила только `30k` шагов против старой speech-базы около `122k` плюс последующего prosody/ranking fine-tune; корпус расширился с `10` до `64` speakers; frame budget упал с `5000` до безопасных `1800`; LR упал с `2e-4` до `2e-5`, а scheduler к концу 30k снизил его ещё примерно до `2e-6`. При грубой оценке новый запуск обработал около `54M` frame-updates против `610M` у старой 122k-базы. Validation подтверждает bottleneck content/acoustic path: новая `CTC=2.4117`, `mel-L1=0.3737`, старая plain 46M на 30k имела `CTC=1.2471`, `mel-L1=0.3391`, а rank-loss checkpoint на 34500 — `CTC=1.0427`, `mel-L1=0.3362`. Отдельный разбор сохранён в `reports/diagnostics/150m_vs_46m_artifact_audit_2026-09-12.md`; просодию до стабилизации acoustic flow подключать не рекомендуется.

В `2026-09-12 13:37` по решению пользователя запущен rescue fine-tune от весов speech-only `step=30000`, но через `--init-checkpoint`, то есть с новым optimizer/scheduler и без старого почти нулевого LR. Архитектура, 64-speaker корпус и speech-only objective сохранены; prosody losses остаются выключенными. Новая ветка использует `learning_rate=5e-5`, warmup `1000`, `frame_budget=1800`, `4` workers и отдельные `checkpoints/transformer_aligned_speech_only_150m_64spk_rescue_30k_latest.pt` / `logs/transformer_aligned_speech_only_150m_64spk_rescue_30k.csv`. Smoke на `3` шагах прошёл с загрузкой `346/346` tensor entries и без OOM; full run жив на `step=400`, GPU/VRAM в контрольном чтении `56%` / `3.41 GB из 8.15 GB`. Исходный `step=30000` checkpoint не перезаписывается.

В `2026-09-12 15:19` rescue fine-tune штатно завершён на `30000/30000`. Validation: `total=0.7019`, `mel-L1=0.3389`, `duration=0.1420`, `speaker=0.000084`, `x0=0.0169`, `CTC=1.7017`; относительно исходной speech-only `step=30000` это примерно `−22%` total, `−9%` mel-L1, `−29%` CTC и `−36%` duration. Сгенерирована контрольная сетка из `12` WAV в `samples/speech_only_150m_rescue_30k_known_text_grid/` с теми же текстами, speaker IDs, seed `42` и `32` flow steps. Первичная слуховая оценка пользователя: основной «рык» почти ушёл, но состояние речи всё ещё плохое. Вывод: новый LR/scheduler действительно улучшил акустическую стабильность, однако нормальная разборчивая речь ещё не достигнута; исходный checkpoint `speech_only_150m_64spk_30k` сохранён отдельно.

В `2026-09-12 15:26` запущен второй отдельный rescue-проход для проверки дальнейшего дообучения: init-веса — `transformer_aligned_speech_only_150m_64spk_rescue_30k_latest.pt`, те же `150M` архитектура, `64` speakers, float32 mel-cache, speech-only objective, `frame_budget=1800`, `learning_rate=5e-5`, warmup `1000`, цель ещё `30000` шагов. Выходы не перезаписывают ни исходную speech-only модель, ни первый rescue: `checkpoints/transformer_aligned_speech_only_150m_64spk_rescue2_30k_latest.pt` и `logs/transformer_aligned_speech_only_150m_64spk_rescue2_30k.csv`. Стартовая загрузка прошла без OOM; первые `250` шагов записаны.

В `2026-09-12 16:12` построен единый timeline для 150M семейства: speech-only с нуля `0–30k`, первый rescue с локальными шагами сдвинут в глобальный диапазон `30–60k`, второй rescue — в `60–77.35k` на момент построения. Ранняя joint+prosody ветка показана пунктиром как параллельная, а не как продолжение speech-only. График: `reports/training_losses/unified_150m_timeline_2026-09-12/all_150m_runs_unified_timeline.webp`; генератор: `scripts/plot_150m_unified_timeline_png.py`.

В `2026-09-12 16:40` проведён кодовый аудит причины устойчиво плохой речи при падающих loss. Выявлен главный риск train/inference mismatch в aligned-модели: в `forward()` `frame_text` строится через CTC forced alignment, причём CTC получает целевую `target_mel` и его путь отсоединён через `ctc_log_probs.detach()`; в `sample()` целевого mel нет, поэтому `frame_text` строится только из предсказанного `duration_head` и эвристического `fit_durations()`. Следовательно, flow во время обучения видит target-зависимое почти-оракульное выравнивание, а при генерации должен работать с собственным приближённым выравниванием. При этом CTC-loss обучает отдельную alignment-ветку и не является прямым доказательством качества text→audio. Дополнительные факторы: ранний CTC forced path шумный, 64-speaker sampler сглаживает крайне неравномерный корпус (Geralt около `15.9 h`, Yennefer около `1.2 h`), а доступный frame-budget даёт существенно меньше acoustic updates, чем старая 46M ветка. Следующая чистая проверка — один checkpoint и три режима frame alignment: predicted duration, MFA duration и target-derived CTC alignment; это отделит проблему duration/train-inference mismatch от собственно flow/acoustic path. Пока rescue2 не объявляется финальным решением.

В `2026-09-12 16:44` второй rescue остановлен на ближайшем сохранённом checkpoint `step=29000` (несохранённые train-строки до `29300` отброшены). Validation: `total=0.6260`, `mel-L1=0.3297`, `duration=0.1234`, `x0=0.0180`, `CTC=1.3746`. Сгенерирована сетка из `12` WAV в `samples/speech_only_150m_rescue2_29k_known_text_grid/` с теми же тестовыми текстами, speaker IDs, seed `42` и `32` flow steps; исходная speech-only и первый rescue сохранены.

Слуховая оценка `2026-09-12 16:47`: основной рык почти исчез, макродлительность фонем выглядит нормальной, но соседние звуки/буквы переходят друг в друга резко, как независимые сегменты. Это уточняет bottleneck: проблема не в общей длине реплики и не в простом duration head, а в межфонемной акустической непрерывности/coarticulation. Текущая модель подаёт в flow кусочно-постоянный `frame_text` через `repeat_interleave()` с жёсткими границами; `path_to_phone_indices()` также назначает CTC blank-кадры ближайшему phone, не создавая мягкого перехода. Основной flow/mel-L1 и x0-loss не штрафуют отдельно ошибку временных производных, поэтому резкие скачки могут оставаться при приемлемом среднем mel-L1. Следующий targeted A/B: сохранить текущий checkpoint, проверить hard vs MFA alignment, затем добавить небольшой masked transition/delta loss на `x0_pred`/mel (без изменения duration objective) и при необходимости заменить hard phone expansion на мягкое overlap/context upsampling.

В `2026-09-12 16:52–16:53` выполнен targeted A/B на checkpoint rescue2 `step=29000`: две реплики (первый train sample Geralt и первый train sample Yennefer), три режима frame alignment (`predicted`, `MFA`, `ctc_oracle`), `32` flow steps. Первичный запуск был отклонён как некорректный для строгого сравнения, поскольку режимы получили разные noise seeds; скрипт исправлен, все три режима каждой реплики пересчитаны с одинаковым seed, поэтому `samples/alignment_modes_rescue2_29k/` содержит чистый A/B. Скрипт: `scripts/diagnose_alignment_modes.py`.

Слуховой результат targeted A/B `2026-09-12 16:57`: у Geralt `MFA` и `ctc_oracle` практически одинаковы, а `predicted` заметно иначе распределяет локальное время — особенно на «близко»; в одном варианте фраза произносится слишком быстро. Это подтверждает, что CTC alignment уже близок к MFA, а слабое место здесь — inference duration head/эвристическое распределение кадров, не сам CTC. У Yennefer три режима почти неразличимы: значит, её плохая разборчивость/неправильный тембр не объясняются выравниванием. Пользователь слышит у всех вариантов хрип и voice profile, похожий на очень старый женский голос вместо Yennefer. Вывод: duration mismatch — локальная проблема Geralt; для Yennefer доминирует speaker/acoustic representation и качество данных/обучения. Низкий speaker-loss не используется как доказательство слышимой идентичности.

В `2026-09-12 17:08` подготовлен и запущен отдельный fine-tune на `30 000` шагов от сохранённого `transformer_aligned_speech_only_150m_64spk_rescue2_30k_latest.pt` (`step=29000`). В один прогон включены три исправления: (1) внешние MFA per-phone durations/frame alignment там, где кеш существует, с CTC fallback для остальных записей; (2) тот же speaker-classification loss с увеличенным весом `0.10→0.50`; (3) masked transition/delta loss на соседних кадрах `x0_pred` против target `x0`, вес `0.10`, чтобы штрафовать резкие межфонемные скачки. Архитектура и число параметров не менялись, prosody objective по-прежнему выключен. MFA-кеш содержит `21 852` файлов; это не весь 64-speaker корпус, поэтому режим намеренно смешанный и не выбрасывает записи без MFA. Изменения: `src/data.py` (`collate_batch_with_optional_duration`), `src/model/transformer_flow_aligned.py` (MFA alignment override и прямой duration path), `scripts/train_stage_a.py` (mask, transition metric/loss, configurable speaker weight), конфиг `configs/transformer_aligned_speech_only_150m_64spk_mfa_transition_speaker_30k.yaml`, runner `scripts/run_aligned_speech_only_150m_64spk_mfa_transition_speaker_30k.ps1`. Smoke на `3` шагах прошёл на CUDA без OOM, checkpoint rescue2 загрузил `346/346` tensor entries; первый полный train-row на `step=50` содержит `transition=0.00870`. Полный лог и checkpoint отдельные: `logs/transformer_aligned_speech_only_150m_64spk_mfa_transition_speaker_30k.csv` и `checkpoints/transformer_aligned_speech_only_150m_64spk_mfa_transition_speaker_30k_latest.pt`; старые веса не перезаписываются.

В `2026-09-12 17:10` новый fine-tune жив на `step=650`; стартовые `346/346` тензоров загружены, train/validation smoke пройден ранее. Скорость по первым минутам около `4.85 step/s`, ориентировочная ETA до `30k` — `~1 ч 40–50 мин` с учётом validation/checkpoint; процесс использует CUDA, private memory около `9.2 GB`, новых OOM нет. Одноразовый снимок GPU показал `55%`, поэтому это не принимается за устойчивую среднюю утилизацию между батчами; безопасный `frame_budget=1800` пока не менялся.

В `2026-09-12 17:31–17:33` подготовлена изолированная инфраструктура под локальный `CosyVoice3` checkout: `C:/Users/miskl/OneDrive/Документы/CosyVoice`. Проверена модель `pretrained_models/Fun-CosyVoice3-0.5B` — на месте `llm.pt`, `flow.pt`, `hift.pt`, ONNX speech-tokenizer/CAMPPlus и Qwen tokenizer. Добавлены metadata-конвертер `scripts/prepare_cosyvoice3_witcher_data.py`, безопасный parquet-runner `scripts/pack_cosyvoice3_witcher_parquet.ps1`, WSL-setup `scripts/setup_cosyvoice3_wsl.sh`, WSL fine-tune runner `scripts/run_cosyvoice3_witcher_finetune_wsl.sh` и описание `experiments/cosyvoice3_witcher/README.md`. Конвертер выполнен без копирования/удаления WAV: выбрано `train=10 219` (`11.28 h`, `59` speakers), `validation=590` (`0.66 h`, `58`), `test=1 817` (`2.01 h`, `64`); отсутствующие записи только отражены в `data/prepare_report.json`. Все четыре filelist-файла в каждом split имеют одинаковое число строк. Официальный parquet пока не создавался: он продублирует около `4.59 GB` WAV и займёт ориентировочно `5.05 GB`; при свободных `37.93 GB` это безопасно, но запуск оставлен отдельным действием. Windows Conda `cosyvoice` пригоден для подготовки/инференса; официальный train импортирует DeepSpeed, поэтому отдельный Linux/WSL venv подготовлен, но не установлен и не запущен. Параллельный Witcher-прогон PID `22568` проверен живым и не прерывался.

В `2026-09-12 20:50–21:57` заново скачаны с GitHub и Steam-DepotDownloader официальные русские voice depots Witcher 3: base `292039`, Hearts of Stone `292047`, Blood and Wine `310616`; суммарный размер скачанных контейнеров около `4.12 GB`. Для извлечения использованы готовые WolvenKit 7.2.0, JTGizmo `w3utils` и официальный `vgmstream-cli`; самописный форматный парсер не добавлялся. Из-за пустых записей в большом `content4` применён готовый Lua extractor hhrhhr с адаптацией только версии контейнера `162→163`. Восстановлено `13 565` русских WAV из `content4` и ещё `22 658` из остальных контейнеров; `13` потоков не нашли текущую metadata-пару. Исходные существующие WAV не перезаписывались.

По запросу пользователя в `2026-09-12 21:44–21:48` удалены только английские `eng.wav` из `dataset_staging` (`21 124` файла, около `7.75 GB`) и польские `pl.wav` (`21 553` файла, около `7.61 GB`). Русские WAV, тексты, metadata и скачанные депоты сохранены. После очистки и восстановления свободно около `15.06 GB`; CosyVoice parquet ещё не создавался.

В `2026-09-12 22:04–22:24` по запросу пользователя удалены временный recovery-кеш `downloads/witcher3_ru_depots/recovered_ru` (`28.03 GB`) и официальные русские депоты base/Hearts of Stone/Blood and Wine (`около 4.12 GB`). `dataset_staging`, проектные скрипты и CosyVoice checkout не удалялись; свободное место выросло примерно до `47.33 GB`, затем после parquet-сборки составило `37.57 GB`.

После восстановления русских WAV повторно подготовлен CosyVoice3 scaffold из `manifests/phoneme_64`: train `30 146` записей (`34.03 h`), validation `1 655` (`1.91 h`), test `1 817` (`2.01 h`), `64` speakers. Упакован официальный CosyVoice parquet-формат с аудиобайтами: `61` train-файл (`10.046 GB`), `4` validation (`0.563 GB`), `4` test (`0.595 GB`), всего `11.203 GB` и `33 618` строк; проверка `pyarrow` прочитала все `69` файлов без ошибок и с точным совпадением числа записей.

Windows-совместимость локального запуска зафиксирована в `scripts/pack_cosyvoice3_witcher_parquet.ps1` и производном `scripts/make_parquet_list_cosyvoice3_windows.py`: официальный `make_parquet_list.py` не изменялся. Исправления только инфраструктурные: lazy-import `torch` (при отсутствии `.pt`-фичей), передача словарей в multiprocessing spawn, явное ожидание ошибок worker-ов и корректное построение speaker-map.

### 2026-09-13 — ClearerVoice-Studio: закрытый эксперимент

Для проверки очистки/разделения речи установлен официальный `modelscope/ClearerVoice-Studio` из GitHub в `C:\witchervoicesdataset\third_party\ClearerVoice-Studio`. Использовалось отдельное conda-окружение `clearvoice` (Python 3.11, ClearVoice 0.1.2, Streamlit, Silero VAD, PyTorch `2.11.0+cu128`); CUDA корректно определялась как NVIDIA GeForce RTX 5050 Laptop GPU. Запускалась официальная Streamlit-морда на `http://127.0.0.1:8501`; предобученные ClearVoice-модели загружаются лениво при первом выборе операции.

Результат пользовательской проверки: остаточные шумы и мелодии отделяются плохо, особенно во время речи; для нашей задачи очистки игровых реплик качество неприемлемо. Эксперимент закрыт, ClearVoice признан неподходящим.

По запросу пользователя остановлены Streamlit-сервер и удалены только созданные для этого эксперимента репозиторий `C:\witchervoicesdataset\third_party\ClearerVoice-Studio`, conda-окружение `clearvoice` и связанные ClearVoice-кеши/веса. CosyVoice, его окружение и веса, `witcher_tts\reports`, `witcher_tts\samples`, исходный датасет и аудиофайлы не удалялись.

### 2026-09-13 — Deepdub Managed Dub: проверка доступа

Для теста выбран локальный `C:\Users\miskl\Downloads\New project.mp4` (`24.29` с, `1920×1080`, `14.39 MB`). По явному разрешению пользователя файл временно загружен в direct-download storage и отправлен в официальный Deepdub endpoint `POST /dubbing/job` с направлением `en-US → ru-RU`, workflow `AUTOMATED` и mix `REPLACEMENT`. API вернул `HTTP 403: Customer is not enabled for Managed Dub`; `requestId` не выдан, поэтому дубляж не создавался и результата нет. Причина — права аккаунта/API-ключа, а не локальный файл или его формат. Временная внешняя копия не является частью датасета; ключ после этой проверки рекомендуется перевыпустить.
