alpha = 0.65  # расчет смещений боксов по x и по y c перекрытием overloapratio=alpha
min_box_area = 1500
max_box_area = 20000  # площади региона
n_sc = 5  # число шагов поиска по масштабу
min_sides_ratio = 1 / 2  # минимальное соотношение сторон (варианты 1/4,1/3,1/2)
delta = 10  # ширина полосы вдоль внешней границы бокса
threshold = 120  # порог баниризации
min_box_ratio = 0.25
max_border_ratio = 0.25
rsort_key = "box_border_ratio"
num_proposal = 2  # количество предложений лучших регионов
# по показателю максимума ib0/S0*(1-ib1/S1)
test_path = 'test_data'
# path_new = 'test_data_new'