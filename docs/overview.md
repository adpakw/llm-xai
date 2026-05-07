# Overview

## Введение и мотивация
> Even years after a large model is trained, both creators and users routinely discover model capabilities – including problematic behaviors – they were previously unaware of.

**Проблемы blackbox'a:**
- Проблема безопасности (Safety):
Модели выдают предвзятые (biased), токсичные или ложные (hallucinated) ответы.
Мы не можем это исправить системно, потому что не знаем, как именно эти ошибки возникают внутри архитектуры. Мы можем только подбирать фильтры на выходе или дообучать на новых данных, действуя наугад.
- ...

для решения этих проблем нужна не просто статистика ошибок, а понимание (mechanistic interpretability) — то есть способность читать внутренние вычисления модели так же, как мы читаем исходный код программы



<!-- записки -->
* анализ attention maps показывает куда смотрит модель, но не почему она принимает решение и как она использует эту информацию. mechanistic interpretability требует восстановления полного алгоритма


https://pair.withgoogle.com/explorables/sae/~