

def relation_im(self, im_path, res):
        boxes_img = res['box']
        pred_cls_img = np.array(res['cls'])
        pred_confs = np.array(res['confs'])
        time1 = time.time()
        im = cv2.imread(im_path)
        ih = im.shape[0]
        iw = im.shape[1]
        PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
        image_blob, im_scale = prep_im_for_blob(im, PIXEL_MEANS)
        blob = np.zeros((1,)+image_blob.shape, dtype=np.float32)
        blob[0] = image_blob        
        # Reshape net's input blobs
        boxes = np.zeros((boxes_img.shape[0], 5))        
        boxes[:, 1:5] = boxes_img * im_scale
        classes = pred_cls_img
        ix1 = []
        ix2 = []
        n_rel_inst = len(pred_cls_img)*(len(pred_cls_img)-1)
        rel_boxes = np.zeros((n_rel_inst, 5))
        SpatialFea = np.zeros((n_rel_inst, 8))
        rel_so_prior = np.zeros((n_rel_inst, 70))
        i_rel_inst = 0
        for s_idx in range(len(pred_cls_img)):
            for o_idx in range(len(pred_cls_img)):
                if(s_idx == o_idx):
                    continue
                ix1.append(s_idx)
                ix2.append(o_idx)
                sBBox = boxes_img[s_idx]
                oBBox = boxes_img[o_idx]
                rBBox = self.getUnionBBox(sBBox, oBBox, ih, iw)
                rel_boxes[i_rel_inst, 1:5] = np.array(rBBox) * im_scale
                SpatialFea[i_rel_inst] = self.getRelativeLoc(sBBox, oBBox)
                rel_so_prior[i_rel_inst] = self.so_prior[classes[s_idx], classes[o_idx]]
                i_rel_inst += 1    
        boxes = boxes.astype(np.float32, copy=False)
        classes = classes.astype(np.float32, copy=False) 
        ix1 = np.array(ix1)
        ix2 = np.array(ix2)    
        obj_score, rel_score = self.net(blob, boxes, rel_boxes, SpatialFea, classes, ix1, ix2, self.args)
        rel_prob = rel_score.data.cpu().numpy()
        rel_prob += np.log(0.5*(rel_so_prior+1.0/self.args.num_relations))        
        rlp_labels_im  = np.zeros((rel_prob.shape[0]*rel_prob.shape[1], 5), dtype = np.int)
        tuple_confs_im = []
        n_idx = 0
        for tuple_idx in range(rel_prob.shape[0]):
            sub = ix1[tuple_idx]            
            obj = ix2[tuple_idx]            
            for rel in range(rel_prob.shape[1]):                
                conf = rel_prob[tuple_idx, rel]
                rlp_labels_im[n_idx] = [classes[sub], sub, rel, classes[obj], obj]
                tuple_confs_im.append(conf)
                n_idx += 1
        tuple_confs_im = np.array(tuple_confs_im)
        idx_order = tuple_confs_im.argsort()[::-1][:20]
        rlp_labels_im = rlp_labels_im[idx_order,:]
        tuple_confs_im = tuple_confs_im[idx_order]
        vrd_res = []
        for tuple_idx in range(rlp_labels_im.shape[0]):
            label_tuple = rlp_labels_im[tuple_idx]
            sub_cls = self.vrd_classes[label_tuple[0]]
            obj_cls = self.vrd_classes[label_tuple[3]]
            rel_cls = self.vrd_rels[label_tuple[2]]
            vrd_res.append(('%s%d-%s-%s%d'%(sub_cls, label_tuple[1], rel_cls, obj_cls, label_tuple[4]), tuple_confs_im[tuple_idx]))        
        print vrd_res
        time2 = time.time()
        print "TEST Time:%s" % (time.strftime('%H:%M:%S', time.gmtime(int(time2 - time1))))
        return vrd_res
