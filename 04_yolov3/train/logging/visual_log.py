from train.logging.metric import split_true_false


class VisualLog:
    def __call__(self, step, grtr, pred):
        """
        :param step:
        :param grtr:
        :param pred:
        :return:
        """
        splits = split_true_false(grtr["bboxes"], pred["nms"])
        # for i in range(batch):
        #     image_grtr = image[i].copy()
        #     image_grtr = draw_boxes(image_grtr, splits["grtr_tp"][i], (0, 255, 0))
        #     image_grtr = draw_boxes(image_grtr, splits["grtr_fn"][i], (0, 0, 255))
        #     image_pred = image[i].copy()
        #     image_pred = draw_boxes(image_pred, splits["pred_tp"][i], (0, 255, 0))
        #     image_pred = draw_boxes(image_pred, splits["pred_fp"][i], (0, 0, 255))
        #     concat
        #     save_file


