import matplotlib.pyplot as plt

def get_cumulative_ranks_augmentation():
    cumulative_ranks_classifier = [229, 246, 252, 285, 327, 354, 371, 397, 416, 419, 429, 437, 452, 456, 457, 462, 463]
    cumulative_ranks_last_conv = [236, 260, 300, 350, 354, 388, 408, 435, 438, 444, 455, 463]
    cumulative_ranks_full = [268, 281, 308, 315, 331, 338, 368, 376, 384, 402, 405, 425, 434, 451, 462, 463]
    cumulative_ranks_simplified = [259, 275, 312, 321, 335, 361, 379, 407, 414, 426, 441, 446, 451, 459, 460, 463]
    cumulative_ranks_svm_linear = [411, 440, 442, 443, 448, 450, 461, 462, 463]
    cumulative_ranks_svm_quadratic = [352, 404, 420, 436, 443, 446, 447, 448, 449, 453, 458, 461, 462, 463]
    cumulative_ranks_svm_exp = [304, 310, 314, 389, 394, 417, 420, 436, 455, 457, 458, 459, 461, 462, 463]

    cumulative_ranks = [
        cumulative_ranks_classifier,
        cumulative_ranks_last_conv,
        cumulative_ranks_full,
        cumulative_ranks_simplified,
        cumulative_ranks_svm_linear,
        cumulative_ranks_svm_quadratic,
        cumulative_ranks_svm_exp
    ]

    return cumulative_ranks

def get_cumulative_ranks_no_augmentation():
    cumulative_ranks_classifier = [258, 266, 300, 309, 338, 361, 389, 406, 413, 436, 440, 443, 456, 461, 462, 463]
    cumulative_ranks_last_conv = [322, 337, 354, 373, 406, 411, 422, 450, 455, 458, 462, 463]
    cumulative_ranks_full = [374, 378, 394, 417, 430, 440, 444, 457, 461, 462, 463]
    cumulative_ranks_simplified = [375, 401, 408, 423, 440, 446, 456, 462, 463]
    cumulative_ranks_svm_linear = [451, 457, 459, 461, 462, 463]
    cumulative_ranks_svm_quadratic = [443, 454, 455, 456, 457, 458, 460, 461, 462, 463]
    cumulative_ranks_svm_exp = [447, 457, 458, 460, 461, 462, 463]

    cumulative_ranks = [
        cumulative_ranks_classifier,
        cumulative_ranks_last_conv,
        cumulative_ranks_full,
        cumulative_ranks_simplified,
        cumulative_ranks_svm_linear,
        cumulative_ranks_svm_quadratic,
        cumulative_ranks_svm_exp
    ]

    return cumulative_ranks


cumulative_ranks = get_cumulative_ranks_augmentation()

plt.figure(figsize=(9,6))


plt.title('Krzywe CMC dla różnych typów sieci - zbiór testowy')
plt.ylim([0, 1.05])
plt.xlim([1, 16])

plt.xlabel('Ranga')
plt.ylabel('Prawdopodobieństwo rangi')

for ranks in cumulative_ranks:
    x = [x for x in range(len(ranks))]

    probs = [x / 463 for x in ranks]
    plt.plot(x, probs, linestyle=':', marker='.', markersize=10)


z = [a for a in range(1, 18)]
plt.xticks(z)

plt.legend(['Klasyfikator', 'Ostatnia splotowa', 'W pełni uczona', 'Uproszczona', 'SVM jądro liniowe', 'SVM jądro kwadratowe', 'SVM jądro wykładnicze'])

plt.savefig('cmc.png', dpi='figure')
