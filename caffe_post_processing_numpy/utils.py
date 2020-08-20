import mmcv
import caffe
import numpy as np

def pre_process_img(one_img, mean, std, to_RGB):

    one_img = mmcv.imnormalize(one_img, mean, std, to_RGB)
    one_img = one_img.transpose(2, 0, 1)
    return one_img

def imread_img(img_path, input_shape, norm_data_cfg):
    
    # read image
    one_img_raw = mmcv.imread(img_path, 'color')
    one_img = mmcv.imresize(one_img_raw, input_shape[1:])

    # img meta_info
    img_meta = dict(
        filename = img_path, 
        ori_filename = img_path.split('/')[-1],
        ori_shape = tuple(one_img_raw.shape),
        img_shape = tuple(one_img.shape),
        pad_shape = (320, 320, 3),
        scale_factor = np.array([1, 1, 1, 1], dtype=np.float32),
        flip = False, 
        flip_direction = 'horizontal',
        img_norm_cfg = norm_data_cfg
    )
    
    return one_img, img_meta

def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        return [bboxes[labels == i, :] for i in range(num_classes)]

def show_result(img,
                result,
                score_thr=0.3,
                bbox_color='green',
                text_color='green',
                thickness=1,
                font_scale=0.5,
                win_name='',
                show=False,
                wait_time=0,
                out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > score_thr)[0]
            np.random.seed(42)
            color_masks = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
            for i in inds:
                i = int(i)
                color_mask = color_masks[labels[i]]
                mask = segms[i]
                img[mask] = img[mask] * 0.5 + color_mask * 0.5
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        mmcv.imshow_det_bboxes(
            img,
            bboxes,
            labels,
            class_names='person',
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            thickness=thickness,
            font_scale=font_scale,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img

