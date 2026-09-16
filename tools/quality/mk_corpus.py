#!/usr/bin/env python3
"""T8.5: deterministic 200-case quality corpus builder (seeded, no LLM).

Writes tools/quality/corpus_t85.json: [{id, category, prompt, max_new,
check:{...}}]. Check types: exact | final_number | unit_test | format |
retrieval | stability | record. Runner renders prompts with the pinned chat
template (thinking off) and feeds ids via --ids-file.
"""
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
cases = []


def add(cat, prompt, max_new, check):
    cases.append({"id": f"{cat[:4]}-{len([c for c in cases if c['category']==cat]):03d}",
                  "category": cat, "prompt": prompt, "max_new": max_new,
                  "check": check})


def factual():
    capitals = [("France", "Paris"), ("Japan", "Tokyo"),                 ("Brazil", "Brasília"),
                ("Canada", "Ottawa"), ("Australia", "Canberra"),
                ("Egypt", "Cairo"), ("India", "New Delhi"),
                ("Germany", "Berlin"), ("Italy", "Rome"),
                ("Spain", "Madrid"), ("Mexico", "Mexico City"),
                ("Argentina", "Buenos Aires")]
    for ctry, cap in capitals:
        add("factual", f"What is the capital of {ctry}? Reply with only the city name.",
            16, {"type": "exact", "expected": cap})
    elements = [("H", "Hydrogen"), ("He", "Helium"), ("C", "Carbon"),
                ("N", "Nitrogen"), ("O", "Oxygen"), ("Na", "Sodium"),
                ("Fe", "Iron"), ("Au", "Gold")]
    for sym, name in elements:
        add("factual", f"Which chemical element has the symbol {sym}? Reply with only the element name.",
            16, {"type": "exact", "expected": name})
    events = [("the first moon landing", "1969"),
              ("the fall of the Berlin Wall", "1989"),
              ("the first modern Olympic Games", "1896"),
              ("the signing of the US Declaration of Independence", "1776"),
              ("the first flight by the Wright brothers", "1903"),
              ("the end of World War II", "1945"),
              ("the launch of Sputnik 1", "1957"),
              ("the invention of the telephone by Bell", "1876")]
    for ev, yr in events:
        add("factual", f"In which year did {ev} occur? Reply with only the year.",
            16, {"type": "exact", "expected": yr})
    conv = [("32 kilometers", "miles", 19.88), ("100 kilometers", "miles", 62.14),
            ("5 kilograms", "pounds", 11.02), ("20 kilograms", "pounds", 44.09),
            ("2 liters", "fluid ounces", 67.63), ("500 milliliters", "fluid ounces", 16.91),
            ("25 degrees Celsius", "Fahrenheit", 77.0), ("0 degrees Celsius", "Fahrenheit", 32.0),
            ("10 meters", "feet", 32.81), ("3 meters", "feet", 9.84),
            ("7 hectares", "acres", 17.3), ("2 hectares", "acres", 4.94)]
    for qty, unit, ans in conv:
        add("factual", f"Convert {qty} to {unit}. Reply with only the number, rounded to two decimals.",
            16, {"type": "final_number", "expected": ans, "tol": 0.05})


def arithmetic(rng):
    for _ in range(6):
        a, b = rng.randint(100, 999), rng.randint(100, 999)
        add("arithmetic", f"What is {a} + {b}? Reply with only the number.",
            16, {"type": "final_number", "expected": a + b, "tol": 0.0})
    for _ in range(4):
        a, b = rng.randint(1000, 9999), rng.randint(100, 999)
        add("arithmetic", f"What is {a} - {b}? Reply with only the number.",
            16, {"type": "final_number", "expected": a - b, "tol": 0.0})
    for _ in range(4):
        a, b = rng.randint(12, 99), rng.randint(12, 99)
        add("arithmetic", f"What is {a} * {b}? Reply with only the number.",
            16, {"type": "final_number", "expected": a * b, "tol": 0.0})
    for _ in range(2):
        b = rng.randint(11, 49)
        q = rng.randint(11, 99)
        add("arithmetic", f"What is {b * q} / {b}? Reply with only the number.",
            16, {"type": "final_number", "expected": q, "tol": 0.0})
    for _ in range(8):
        a, b, c = rng.randint(10, 99), rng.randint(10, 99), rng.randint(2, 12)
        add("arithmetic", f"What is ({a} + {b}) * {c}? Reply with only the number.",
            16, {"type": "final_number", "expected": (a + b) * c, "tol": 0.0})
    for _ in range(8):
        n = rng.randint(6, 30)
        p = rng.choice([5, 10, 15, 20, 25])
        add("arithmetic", f"A shop sells {n} apples per day. How many apples does it sell in {p} days? Reply with only the number.",
            16, {"type": "final_number", "expected": n * p, "tol": 0.0})
    for _ in range(8):
        a, b = rng.randint(50, 500), rng.randint(7, 48)
        add("arithmetic", f"What is the remainder of {a} divided by {b}? Reply with only the number.",
            16, {"type": "final_number", "expected": a % b, "tol": 0.0})


def code(prompt, tests):
    add("coding", prompt, 256,  # 256: codi-014 truncated at 128 (T8.5)
        {"type": "unit_test", "tests": tests})


def coding():
    T = [
        ("Write a Python function add(a, b) that returns their sum. Reply with only the function code, no explanation.",
         "assert add(2, 3) == 5\nassert add(-1, 1) == 0\nassert add(0, 0) == 0"),
        ("Write a Python function mul(a, b) that returns their product. Reply with only the function code, no explanation.",
         "assert mul(3, 4) == 12\nassert mul(-2, 5) == -10"),
        ("Write a Python function is_even(n) that returns True if n is even else False. Reply with only the function code.",
         "assert is_even(4) is True\nassert is_even(7) is False\nassert is_even(0) is True"),
        ("Write a Python function factorial(n) that returns n! for n >= 0. Reply with only the function code.",
         "assert factorial(0) == 1\nassert factorial(5) == 120\nassert factorial(7) == 5040"),
        ("Write a Python function fib(n) that returns the n-th Fibonacci number with fib(0)=0, fib(1)=1. Reply with only the function code.",
         "assert fib(0) == 0\nassert fib(1) == 1\nassert fib(10) == 55"),
        ("Write a Python function reverse_string(s) that returns s reversed. Reply with only the function code.",
         "assert reverse_string('abc') == 'cba'\nassert reverse_string('') == ''"),
        ("Write a Python function is_palindrome(s) that returns True if s reads the same forwards and backwards. Reply with only the function code.",
         "assert is_palindrome('racecar') is True\nassert is_palindrome('hello') is False"),
        ("Write a Python function count_vowels(s) that returns the number of vowels (aeiou, case-insensitive) in s. Reply with only the function code.",
         "assert count_vowels('hello') == 2\nassert count_vowels('AEIOU') == 5\nassert count_vowels('xyz') == 0"),
        ("Write a Python function sum_list(xs) that returns the sum of a list of numbers. Reply with only the function code.",
         "assert sum_list([1, 2, 3]) == 6\nassert sum_list([]) == 0"),
        ("Write a Python function list_max(xs) that returns the largest element of a non-empty list. Reply with only the function code.",
         "assert list_max([3, 1, 4, 1, 5]) == 5\nassert list_max([-2, -8]) == -2"),
        ("Write a Python function second_largest(xs) that returns the second largest distinct value in a list. Reply with only the function code.",
         "assert second_largest([3, 1, 4, 1, 5]) == 4\nassert second_largest([10, 10, 5]) == 5"),
        ("Write a Python function dedup(xs) that returns a list with duplicates removed, preserving order. Reply with only the function code.",
         "assert dedup([1, 2, 2, 3, 1]) == [1, 2, 3]\nassert dedup([]) == []"),
        ("Write a Python function flatten(xss) that flattens a list of lists into one list. Reply with only the function code.",
         "assert flatten([[1, 2], [3], []]) == [1, 2, 3]"),
        ("Write a Python function dot(a, b) that returns the dot product of two equal-length vectors. Reply with only the function code.",
         "assert dot([1, 2, 3], [4, 5, 6]) == 32"),
        ("Write a Python function mat2x2_mul(A, B) that multiplies two 2x2 matrices given as nested lists. Reply with only the function code.",
         "assert mat2x2_mul([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[19, 22], [43, 50]]"),
        ("Write a Python function gcd(a, b) that returns the greatest common divisor. Reply with only the function code.",
         "assert gcd(12, 18) == 6\nassert gcd(7, 13) == 1\nassert gcd(100, 25) == 25"),
        ("Write a Python function is_prime(n) that returns True if n is prime. Reply with only the function code.",
         "assert is_prime(2) is True\nassert is_prime(15) is False\nassert is_prime(29) is True"),
        ("Write a Python function primes_upto(n) that returns all primes <= n in order. Reply with only the function code.",
         "assert primes_upto(10) == [2, 3, 5, 7]\nassert primes_upto(2) == [2]"),
        ("Write a Python function binary_search(xs, x) that returns the index of x in sorted xs or -1. Reply with only the function code.",
         "assert binary_search([1, 3, 5, 7], 5) == 2\nassert binary_search([1, 3, 5], 4) == -1"),
        ("Write a Python function bubble_pass(xs) that performs one bubble-sort pass over the list in place and returns it. Reply with only the function code.",
         "assert bubble_pass([3, 2, 1]) == [2, 1, 3]"),
        ("Write a Python function merge_sorted(a, b) that merges two sorted lists into one sorted list. Reply with only the function code.",
         "assert merge_sorted([1, 3], [2, 4]) == [1, 2, 3, 4]\nassert merge_sorted([], [1]) == [1]"),
        ("Write a Python function word_count(s) that returns a dict mapping each word to its count (split on spaces). Reply with only the function code.",
         "assert word_count('a b a') == {'a': 2, 'b': 1}"),
        ("Write a Python function most_common(xs) that returns the most frequent element. Reply with only the function code.",
         "assert most_common([1, 2, 2, 3]) == 2"),
        ("Write a Python function group_by_first(words) that groups words by their first letter into a dict. Reply with only the function code.",
         "assert group_by_first(['apple', 'ape', 'boat']) == {'a': ['apple', 'ape'], 'b': ['boat']}"),
        ("Write a Python function fizzbuzz(n) that returns a list of strings for 1..n with Fizz/Buzz/FizzBuzz rules. Reply with only the function code.",
         "assert fizzbuzz(5) == ['1', '2', 'Fizz', '4', 'Buzz']\nassert fizzbuzz(15)[-1] == 'FizzBuzz'"),
        ("Write a Python function running_total(xs) that returns the running totals of a list. Reply with only the function code.",
         "assert running_total([1, 2, 3]) == [1, 3, 6]\nassert running_total([]) == []"),
        ("Write a Python function clamp(x, lo, hi) that restricts x to the range [lo, hi]. Reply with only the function code.",
         "assert clamp(5, 0, 10) == 5\nassert clamp(-3, 0, 10) == 0\nassert clamp(99, 0, 10) == 10"),
        ("Write a Python function celsius_to_fahrenheit(c) that converts Celsius to Fahrenheit. Reply with only the function code.",
         "assert abs(celsius_to_fahrenheit(0) - 32.0) < 1e-9\nassert abs(celsius_to_fahrenheit(100) - 212.0) < 1e-9"),
        ("Write a Python function char_freq(s) that returns a dict of character frequencies. Reply with only the function code.",
         "assert char_freq('aab') == {'a': 2, 'b': 1}"),
        ("Write a Python function transpose(matrix) that transposes a rectangular nested list. Reply with only the function code.",
         "assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]"),
    ]
    assert len(T) == 30
    for prompt, tests in T:
        code(prompt, tests)


def summarization():
    paras = [
        "The harbor seal rested on the warm rocks while gulls circled overhead. Fishermen mended their nets nearby, and the tide slowly pulled the boats away from the dock. By evening the water was calm again.",
        "Photosynthesis allows green plants to convert sunlight, water, and carbon dioxide into glucose and oxygen. Chlorophyll captures light energy, which powers chemical reactions in the leaves. This process underpins most life on Earth.",
        "The old library on Baker Street holds over forty thousand volumes. Its reading room has tall oak shelves, brass lamps, and a marble staircase. Scholars travel from far away to consult its rare manuscript collection.",
        "A thunderstorm rolled over the valley just after midnight. Lightning split a tall pine on the ridge, and rain hammered the tin roof of the cabin. By dawn only wet leaves and a fresh smell of pine remained.",
        "The bridge spans the river in three steel arches. Engineers spent four years driving piles into the riverbed before the first girder was lifted. Today thousands of cars cross it every hour.",
        "Bees communicate the location of flowers through a waggle dance. The angle of the dance relative to the sun indicates direction, while its duration signals distance. Other workers watch and then fly straight to the source.",
        "The museum's new wing displays pottery from five ancient cultures. Each vessel is labeled with its origin, estimated age, and firing technique. Interactive screens let visitors rotate fragile pieces virtually.",
        "Winter wheat is planted in autumn and harvested in early summer. The young plants survive frost under a blanket of snow, then grow rapidly when temperatures rise. Farmers watch rainfall closely in April and May.",
        "The lighthouse keeper climbed one hundred twelve steps each evening. He polished the lens, lit the lamp, and logged passing ships. On foggy nights he also sounded the horn every two minutes.",
        "Coral reefs grow where warm, shallow, sunlit water meets a rocky shelf. Tiny polyps build limestone skeletons over centuries, creating habitats for thousands of species. Rising temperatures bleach and weaken them.",
    ]
    assert len(paras) == 10
    for i, p in enumerate(paras):
        add("summarization", f"Summarize the following paragraph in one sentence:\n\n{p}",
            96, {"type": "format", "max_sentences": 2, "min_chars": 20})
        add("summarization", f"Rewrite the following paragraph in a formal style, keeping all facts:\n\n{p}",
            128, {"type": "format", "min_chars": 60})


def bilingual():
    en = [("The cat sat on the mat.", "The cat sat on the mat."),
          ("Water boils at 100 degrees Celsius.", "Water boils at 100 degrees Celsius."),
          ("She sells seashells by the seashore.", "She sells seashells by the seashore."),
          ("The quick brown fox jumps over the lazy dog.", "The quick brown fox jumps over the lazy dog."),
          ("	breaking news: markets rally on strong earnings.  ".strip(), "breaking news: markets rally on strong earnings."),
          ("An apple a day keeps the doctor away.", "An apple a day keeps the doctor away."),
          ("To be or not to be, that is the question.", "To be or not to be, that is the question."),
          ("The early bird catches the worm.", "The early bird catches the worm."),
          ("A journey of a thousand miles begins with a single step.", "A journey of a thousand miles begins with a single step."),
          ("Honesty is the best policy.", "Honesty is the best policy."),
          ("Time flies like an arrow.", "Time flies like an arrow."),
          ("The pen is mightier than the sword.", "The pen is mightier than the sword."),
          ("When in Rome, do as the Romans do.", "When in Rome, do as the Romans do."),
          ("No pain, no gain.", "No pain, no gain."),
          ("Knowledge is power.", "Knowledge is power.")]
    for src, exp in en:
        add("bilingual", f"Repeat the following sentence exactly:\n\n{src}",
            48, {"type": "exact", "expected": exp})
    zh = [("把下面的中文翻译成英文，只回复译文：猫坐在垫子上。", "The cat is sitting on the mat."),
          ("把下面的中文翻译成英文，只回复译文：水在100摄氏度沸腾。", "Water boils at 100 degrees Celsius."),
          ("把下面的中文翻译成英文，只回复译文：今天天气很好。", "The weather is very nice today."),
          ("把下面的中文翻译成英文，只回复译文：我爱读书。", "I love reading."),
          ("把下面的中文翻译成英文，只回复译文：北京是中国的首都。", "Beijing is the capital of China."),
          ("Translate the following English into Chinese, reply with only the translation: Good morning.",
           "早上好。"),
          ("Translate the following English into Chinese, reply with only the translation: Thank you very much.",
           "非常感谢。"),
          ("Translate the following English into Chinese, reply with only the translation: See you tomorrow.",
           "明天见。"),
          ("Translate the following English into Chinese, reply with only the translation: The book is on the table.",
           "书在桌子上。"),
          ("Translate the following English into Chinese, reply with only the translation: I have two apples.",
           "我有两个苹果。"),
          ("下面这句话有多少个汉字？只回复数字：人工智能正在改变世界", "10"),
          ("下面这句话有多少个汉字？只回复数字：春眠不觉晓", "5"),
          ("把下面的句子改成疑问句，只回复改写后的句子：他喜欢喝茶。", "他喜欢喝茶吗？"),
          ("用“因为……所以……”造一个句子。", "record"),
          ("请用三个词语形容大海。", "record")]
    for src, exp in zh:
        if exp == "record":
            add("bilingual", src, 64, {"type": "record"})
        else:
            add("bilingual", src, 48, {"type": "exact", "expected": exp})


def longgen():
    prompts = [
        "Write a short story about a lighthouse keeper who discovers a hidden room.",
        "Explain how rain forms, step by step, for a curious ten-year-old.",
        "Describe a journey through a desert at night in vivid detail.",
        "Write a dialogue between a robot and its inventor about freedom.",
        "Tell the history of writing systems from cuneiform to the alphabet.",
        "Describe how to bake sourdough bread from starter to finished loaf.",
        "Write a story about two friends who build a treehouse over one summer.",
        "Explain the water cycle and why it matters for farmers.",
        "Imagine a city floating on the ocean and describe daily life there.",
        "Write about a musician who hears a melody that nobody else can hear.",
        "Describe the seasons in a mountain valley across one full year.",
        "Explain how bridges stay up, with examples of three bridge types.",
        "Write a mystery story set on an overnight train.",
        "Describe the process of making paper from trees to notebook.",
        "Tell a fable about a fox and a crow with a clear moral.",
        "Write about the first humans landing on Mars and their first week.",
        "Explain electricity to someone who has never seen a light bulb.",
        "Describe a coral reef and the creatures that live there.",
        "Write a story about a library where books rewrite themselves.",
        "Explain photosynthesis and why forests are called the lungs of Earth.",
    ]
    assert len(prompts) == 20
    for p in prompts:
        add("longgen", p, 300, {"type": "stability"})


def retrieval():
    # 5 legacy 64K single-needle variants are REUSED from needle_chain results
    # (no rerun): variants 0..4, codes in tools/t74/needle_corpus.json.
    for v in range(5):
        add("retrieval", f"LEGACY-64K-NEEDLE variant {v} (see tools/t74/report_needle.json)",
            16, {"type": "record", "legacy_variant": v})
    # 15 new shorter-context retrieval cases: generated by mk_retrieval.py
    # (depth x distractor x single/multi-needle at 4K/8K/16K/32K).
    cfgs = [("4K", 4096, 0.1), ("4K", 4096, 0.9), ("8K", 8192, 0.1),
            ("8K", 8192, 0.5), ("8K", 8192, 0.9), ("16K", 16384, 0.1),
            ("16K", 16384, 0.5), ("16K", 16384, 0.9), ("32K", 32768, 0.25),
            ("32K", 32768, 0.75), ("8K-multi", 8192, 0.5),
            ("16K-multi", 16384, 0.5), ("4K-distract", 4096, 0.5),
            ("16K-distract", 16384, 0.5), ("32K-distract", 32768, 0.5)]
    for tag, ctx, depth in cfgs:
        add("retrieval", f"RETRIEVAL-GEN {tag} ctx={ctx} depth={depth}",
            24, {"type": "record", "gen": {"tag": tag, "ctx": ctx, "depth": depth}})


def main():
    rng = random.Random(20260916)
    factual()
    arithmetic(rng)
    coding()
    summarization()
    bilingual()
    longgen()
    retrieval()
    # coding / summarization / bilingual / longgen / retrieval appended below
    out = os.path.join(HERE, "corpus_t85.json")
    json.dump({"version": 1, "seed": 20260916, "cases": cases},
              open(out, "w"), indent=1, ensure_ascii=False)
    import collections
    print("cases so far:", dict(collections.Counter(c["category"] for c in cases)))


if __name__ == "__main__":
    main()
