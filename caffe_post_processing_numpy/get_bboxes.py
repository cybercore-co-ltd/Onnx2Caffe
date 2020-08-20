from bbox_nms import multiclass_nms_caffe
import numpy as np
from anchor_generator import AnchorGenerator

class Get_Bboxes_Caffe(object):
    def __init__(self, strides, ratios, octave_base_scale=8, scales_per_octave=1):

        self.anchor_generator = AnchorGenerator(strides, ratios, octave_base_scale=8, scales_per_octave=1)

    def get_bboxes(self,
                    cls_scores,
                    bbox_preds,
                    img_metas,
                    cfg=None,
                    rescale=False):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                    Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                    level with shape (N, num_anchors * 4, H, W)
                img_metas (list[dict]): Meta information of each image, e.g.,
                    image size, scaling factor, etc.
                cfg (mmcv.Config | None): Test / postprocessing configuration,
                    if None, test_cfg would be used
                rescale (bool): If True, return boxes in original image space.
                    Default: False.

        Returns:
                list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                    The first item is an (n, 5) tensor, where the first 4 columns
                    are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                    5-th column is a score between 0 and 1. The second item is a
                    (n,) tensor where each item is the predicted class labelof the
                    corresponding box.

        Example:
                >>> import mmcv
                >>> self = AnchorHead(
                >>>     num_classes=9,
                >>>     in_channels=1,
                >>>     anchor_generator=dict(
                >>>         type='AnchorGenerator',
                >>>         scales=[8],
                >>>         ratios=[0.5, 1.0, 2.0],
                >>>         strides=[4,]))
                >>> img_metas = [{'img_shape': (32, 32, 3), 'scale_factor': 1}]
                >>> cfg = mmcv.Config(dict(
                >>>     score_thr=0.00,
                >>>     nms=dict(type='nms', iou_thr=1.0),
                >>>     max_per_img=10))
                >>> feat = torch.rand(1, 1, 3, 3)
                >>> cls_score, bbox_pred = self.forward_single(feat)
                >>> # note the input lists are over different levels, not images
                >>> cls_scores, bbox_preds = [cls_score], [bbox_pred]
                >>> result_list = self.get_bboxes(cls_scores, bbox_preds,
                >>>                               img_metas, cfg)
                >>> det_bboxes, det_labels = result_list[0]
                >>> assert len(result_list) == 1
                >>> assert det_bboxes.shape[1] == 5
                >>> assert len(det_bboxes) == len(det_labels) == cfg.max_per_img
        """

        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device='cpu')

        result_list = []

        cls_score_list = [
                cls_scores[i][0] for i in range(num_levels)]

        bbox_pred_list = [
                bbox_preds[i][0] for i in range(num_levels)]

        img_shape = img_metas['img_shape']
        scale_factor = img_metas['scale_factor']

        proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                mlvl_anchors, img_shape,
                                                scale_factor, cfg, rescale)
        result_list.append(proposals)
        return result_list

    def distance2bbox(self, points, distance, max_shape=None):
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom).
            max_shape (tuple): Shape of the image.

        Returns:
            Tensor: Decoded bboxes.
        """
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, a_min=0, a_max=max_shape[1])
            y1 = np.clip(y1, a_min=0, a_max=max_shape[0])
            x2 = np.clip(x2, a_min=0, a_max=max_shape[1])
            y2 = np.clip(y2, a_min=0, a_max=max_shape[0])
        return np.stack([x1, y1, x2, y2], -1)

    def anchor_center(self, anchors):
        """Get anchor centers from anchors.

        Args:
            anchors (Tensor): Anchor list with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Anchor centers with shape (N, 2), "xy" format.
        """
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        return np.stack([anchors_cx, anchors_cy], axis=-1)

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into labeled boxes.
        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                has shape (num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for a single
                scale level with shape (4*(n+1), H, W), n is max value of
                integral set.
            mlvl_anchors (list[Tensor]): Box reference for a single scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): Bbox predictions in shape (N, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (N,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, stride, anchors in zip(
                cls_scores, bbox_preds, self.anchor_generator.strides,
                mlvl_anchors):
            assert cls_score.shape[-2:] == bbox_pred.shape[-2:]
            assert stride[0] == stride[1]

            scores = cls_score.transpose(1, 2, 0).reshape(
                -1, 1)
            scores = 1/(1+np.exp(-scores))

            bbox_pred = bbox_pred.transpose(
                1, 2, 0).reshape(-1, 4) * stride[0]

            nms_pre = cfg.get('nms_pre', -1)
            max_scores = scores.max(axis=1)

            # topk bboxes
            if nms_pre > 0 and scores.shape[0] > nms_pre:  
                topk_inds = np.argpartition(max_scores, -nms_pre)[-nms_pre:]
            else:
                topk_inds = np.argpartition(max_scores, -scores.shape[0])[-scores.shape[0]:]

            anchors = anchors[topk_inds, :]
            bbox_pred = bbox_pred[topk_inds, :]
            scores = scores[topk_inds, :]
            print(img_shape)
            
            bboxes = self.distance2bbox(
                self.anchor_center(anchors), bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = np.concatenate(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= scale_factor

        mlvl_scores = np.concatenate(mlvl_scores)
        # Add a dummy background class to the backend when using sigmoid
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = np.zeros(mlvl_scores.shape)
        mlvl_scores = np.concatenate([mlvl_scores, padding], axis=1)

        det_bboxes, det_labels = multiclass_nms_caffe(mlvl_bboxes, mlvl_scores,
                                                cfg['score_thr'], cfg['nms'],
                                                cfg['max_per_img'])
        return det_bboxes, det_labels