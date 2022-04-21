from pathlib import Path

sample_size = (100, 150)
alpha = 0.65  # расчет смещений боксов по x и по y c перекрытием overloapratio=alpha
min_box_area = 5000
max_box_area = 25000  # площади региона
n_box_sides_steps = 5  # число шагов поиска по масштабу
min_sides_ratio = 1 / 2  # минимальное соотношение сторон (варианты 1/4,1/3,1/2)
delta = 10  # ширина полосы вдоль внешней границы бокса
threshold = 120  # порог баниризации
min_box_ratio =0.5 #0.25
max_border_ratio = 0.25
random_state=1


rsort_key = "box_border_ratio"
test_path = Path('data/test')
barcodes_path = Path("C:/Users/zgstv/OneDrive/Изображения/barcodes_full")
bottles_imgs_path = Path("C:/Users/zgstv/OneDrive/Изображения/vend_machines")
data_path = Path('data')
img_data_path = data_path / "img_data"
train_data_path = data_path / "train_data"
valid_data_path = data_path / "results"
