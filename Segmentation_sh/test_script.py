from Segmentation_sh.test_methods import find_barcode


# @benchmark
def main():
    # test_dir = os.path.normpath("C:/Users/zgstv/OneDrive/Изображения/vend_machines")
    # test_dir = os.path.normpath("C:/Users/zgstv/OneDrive/Изображения/vend_machines2")
    # test_dir = os.path.abspath("test_data")
    # result_dir = os.path.abspath("results/binary_img/v")
    # result_dir = os.path.abspath("results/binary_img/v2")
    # result_dir = os.path.abspath("results/test")
    # test_from_dir(test_dir, result_dir)
    find_barcode("test_data/2021-10-20_13_32_07_783.png")


    # load = True
    # positive_filename = "positive.npy"
    # negative_filename = "negative.npy"
    # if load and os.path.exists(positive_filename) and os.path.exists(negative_filename):
    #     positive = np.load(positive_filename)
    #     negative = np.load(negative_filename)
    # else:
    #     barcodes, barcodes_names = get_images_from_dir(Path("C:/Users/zgstv/OneDrive/Изображения/barcodes_full"))
    #     images, imgs_names = get_images_from_dir(Path("C:/Users/zgstv/OneDrive/Изображения/vend_machines"), barcodes_names)
    #     configs = get_boxes_configurations(min_box_area, max_box_area, n_box_sides_steps, min_sides_ratio)
    #     patch_sizes = configs + [(height, width) for width, height in configs]
    #     count_patches = 10
    #     negative_patches = []
    #     negative_patches.extend(
    #         it.chain.from_iterable((extract_patches(img, patch_sizes, count_patches) for img in images)))
    #
    #     # провести аугментацию штрихкодов ( поворот по вертикали, горизонтали, и так, и так; поворот на несколько
    #     # градусов по часовой и против часовой
    #     sample_size = (100, 150)
    #     positive = np.array([resize_img(make_horizontal(barcode, sample_size), sample_size) for barcode in barcodes])
    #     negative = np.array([resize_img(make_horizontal(patch, sample_size), sample_size) for patch in negative_patches])
    #     assert positive.shape[1:] == negative.shape[1:]
    #     np.save("positive",positive)
    #     np.save("negative",negative)
    #
    # X_train = np.array([feature.hog(im,multichannel=True)
    #                     for im in tqdm(it.chain(positive,
    #                                             negative))])
    #
    # y_train = np.zeros(X_train.shape[0])
    # y_train[:positive.shape[0]] = 1



    # grid = GridSearchCV(LinearSVC(dual=False), {'C': [1.0, 2.0, 4.0, 8.0]}, cv=3)
    # grid.fit(X_train, y_train)
    # print(grid.best_score_)
    # model = grid.best_estimator_
    # X_train_filename = "X_train.npy"
    # y_train_filename = "y_train.npy"
    # X_train = np.load(X_train_filename)
    # y_train = np.load(y_train_filename)
    #
    # model = LinearSVC(C=4.0, dual=False)
    # model.fit(X_train, y_train)

    print()


if __name__ == '__main__':
    main()
