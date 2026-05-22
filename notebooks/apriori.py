from typing import List, Dict, Tuple, Union
import itertools
from collections import defaultdict


def apriori_gen(Lk_minus_1: List[Tuple[str, ...]], k: int) -> set:
    candidates = set()
    L_set = set(Lk_minus_1)

    for i, c1 in enumerate(Lk_minus_1):
        for c2 in Lk_minus_1[i+1:]:
            if c1[:-1] == c2[:-1]:
                candidate = c1 + (c2[-1],)
                # Проверка всех (k-1)-подмножеств
                if all(candidate[:idx] + candidate[idx+1:] in L_set for idx in range(k)):
                    candidates.add(candidate)
    return candidates


def apriori(transactions: List[Union[List, set]], min_support: float) -> Dict[Tuple[str, ...], float]:
    t_sets = [set(t) for t in transactions]
    n = len(t_sets)

    item_counts = defaultdict(int)
    for t in t_sets:
        for item in t:
            item_counts[item] += 1

    frequent_itemsets = {}
    Lk = []
    for item, count in item_counts.items():
        supp = count / n
        if supp >= min_support:
            itemset = (item,)
            Lk.append(itemset)
            frequent_itemsets[itemset] = supp

    Lk.sort()
    k = 2

    while Lk:
        candidates = apriori_gen(Lk, k)
        if not candidates:
            break

        candidate_items = list(candidates)
        candidate_sets = [set(c) for c in candidate_items]

        counts = defaultdict(int)
        for t in t_sets:
            for idx in range(len(candidate_items)):
                if candidate_sets[idx].issubset(t):
                    counts[candidate_items[idx]] += 1

        new_Lk = []
        for c, count in counts.items():
            supp = count / n
            if supp >= min_support:
                new_Lk.append(c)
                frequent_itemsets[c] = supp

        Lk = sorted(new_Lk)
        k += 1

    return frequent_itemsets


def generate_rules(
        frequent_itemsets: Dict[Tuple[str, ...], float],
        min_confidence: float,
        min_lift: float = 1.0,
        max_antecedent_len: int = None,
        max_consequent_len: int = None
) -> List[Dict]:
    rules = []

    for itemset, supp_xy in frequent_itemsets.items():
        if len(itemset) < 2:
            continue

        freq_itemsets_local = frequent_itemsets

        for i in range(1, len(itemset)):
            for X in itertools.combinations(itemset, i):
                Y = tuple(sorted(set(itemset) - set(X)))

                if max_antecedent_len and len(X) > max_antecedent_len:
                    continue
                if max_consequent_len and len(Y) > max_consequent_len:
                    continue

                supp_x = freq_itemsets_local[X]
                supp_y = freq_itemsets_local[Y]
                confidence = supp_xy / supp_x
                lift = supp_xy / (supp_x * supp_y)
                leverage = supp_xy - (supp_x * supp_y)
                conviction = float('inf') if confidence == 1 else (1 - supp_y) / (1 - confidence)

                if confidence >= min_confidence and lift >= min_lift:
                    rules.append({
                        'antecedent': X,
                        'consequent': Y,
                        'support': round(supp_xy, 4),
                        'confidence': round(confidence, 4),
                        'lift': round(lift, 4),
                        'conviction': round(conviction, 4) if conviction != float('inf') else float('inf'),
                        'leverage': round(leverage, 4)
                    })

    return sorted(rules, key=lambda r: r['confidence'], reverse=True)