import numpy as np
import math



class black_box_benchmarks(object):
    
    def __init__(self, shadow_train_performance, shadow_test_performance, 
                 target_train_performance, target_test_performance, num_classes,length = 0):
        '''
        each input contains both model predictions (shape: num_data*num_classes) and ground-truth labels. 
        '''

        self.length = length
        print(self.length)
        self.num_classes = num_classes



        self.s_tr_outputs, self.s_tr_labels = shadow_train_performance
        self.s_te_outputs, self.s_te_labels = shadow_test_performance
        self.t_tr_outputs, self.t_tr_labels = target_train_performance
        self.t_te_outputs, self.t_te_labels = target_test_performance

        self.num_classes = len(self.s_tr_outputs[0])
        #print(f'{self.s_tr_labels=}')

        #
        # print('shadow_train_labels: {}'.format(self.s_tr_labels[0]))
        # print('shadow_train_labels: {}'.format(self.s_tr_labels[0]))
        # print('target_train_output: {}'.format(self.t_tr_outputs[1]))
        #
        # print('target_train_labels: {}'.format(self.t_tr_labels[1]))

        #出索引所以看的是预测是否准确
        # print(type(np.argmax(self.s_tr_outputs, axis=1)))
        # print(f'{len(np.argmax(self.s_tr_outputs, axis=1))=}')
        # print(np.argmax(self.s_tr_outputs, axis=1))
        #print(f'{len(self.t_tr_labels)=}')
        # print(self.s_tr_labels)
        # print(np.argmax(self.s_tr_outputs, axis=1)==self.s_tr_labels)
        self.s_tr_corr = (np.argmax(self.s_tr_outputs, axis=1)==self.s_tr_labels).astype(int)
        #print(self.s_tr_corr)
        self.s_te_corr = (np.argmax(self.s_te_outputs, axis=1)==self.s_te_labels).astype(int)
        self.t_tr_corr = (np.argmax(self.t_tr_outputs, axis=1)==self.t_tr_labels).astype(int)
        self.t_te_corr = (np.argmax(self.t_te_outputs, axis=1)==self.t_te_labels).astype(int)




        #没看懂干嘛
        self.s_tr_conf = np.array([self.s_tr_outputs[i, self.s_tr_labels[i].astype(int)] for i in range(len(self.s_tr_labels))])
        self.s_te_conf = np.array([self.s_te_outputs[i, self.s_te_labels[i].astype(int)] for i in range(len(self.s_te_labels))])
        self.t_tr_conf = np.array([self.t_tr_outputs[i, self.t_tr_labels[i].astype(int)] for i in range(len(self.t_tr_labels))])
        self.t_te_conf = np.array([self.t_te_outputs[i, self.t_te_labels[i].astype(int)] for i in range(len(self.t_te_labels))])


        self.s_tr_entr = self._entr_comp(self.s_tr_outputs)
        self.s_te_entr = self._entr_comp(self.s_te_outputs)
        self.t_tr_entr = self._entr_comp(self.t_tr_outputs)
        self.t_te_entr = self._entr_comp(self.t_te_outputs)



        self.s_tr_m_entr = self._m_entr_comp(self.s_tr_outputs, self.s_tr_labels)
        self.s_te_m_entr = self._m_entr_comp(self.s_te_outputs, self.s_te_labels)
        self.t_tr_m_entr = self._m_entr_comp(self.t_tr_outputs, self.t_tr_labels)
        self.t_te_m_entr = self._m_entr_comp(self.t_te_outputs, self.t_te_labels)




    #setting minimum value to prevent 0 be denominator
    def _log_value(self, probs, small_value=1e-30):
        return -np.log(np.maximum(probs, small_value))

    #cross_entropy
    def _entr_comp(self, probs):
        return np.sum(np.multiply(probs, self._log_value(probs)),axis=1)


    def _m_entr_comp(self, probs, true_labels):
        log_probs = self._log_value(probs)
        reverse_probs = 1-probs
        log_reverse_probs = self._log_value(reverse_probs)
        modified_probs = np.copy(probs)
        modified_probs[range(true_labels.size), true_labels.astype(int)] = reverse_probs[range(true_labels.size), true_labels.astype(int)]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(true_labels.size), true_labels.astype(int)] = log_probs[range(true_labels.size), true_labels.astype(int)]
        return np.sum(np.multiply(modified_probs, modified_log_probs),axis=1)

    #how to setting threshold
    def _thre_setting(self, tr_values, te_values):
        value_list = np.concatenate((tr_values, te_values))
        thre, max_acc = 0, 0
        for value in value_list:
            if(len(tr_values)==0 or len(te_values)==0):
                pass
            else:
                #num of member which score > threshold
                tr_ratio = np.sum(tr_values>=value)/(len(tr_values)+0.0)
                te_ratio = np.sum(te_values<value)/(len(te_values)+0.0)
                acc = 0.5*(tr_ratio + te_ratio)
                if acc > max_acc:
                    thre, max_acc = value, acc
        return thre
    
    def _mem_inf_via_corr(self):
        # perform membership inference attack based on whether the input is correctly classified or not
        print(np.sum(self.t_tr_corr))
        t_tr_acc = np.sum(self.t_tr_corr)/(len(self.t_tr_corr)+0.0)
        t_te_acc = np.sum(self.t_te_corr)/(len(self.t_te_corr)+0.0)
        TP = t_tr_acc*(len(self.t_tr_corr)+0.0)
        FP = t_te_acc*(len(self.t_te_corr)+0.0)
        FN = (len(self.t_tr_corr)+0.0)-TP
        TN = (len(self.t_te_corr)+0.0)-FP


        print('TP:{TP}, FP:{FP}, FN:{FN},TN={TN}'.format(TP=TP, FP=FP, FN=FN,TN=TN))

        R = TP/(TP+FN)
        P = TP/(TP+FP)
        F1 = 2*P*R/(P+R)
        FPR = FP/(FP+TN)
        MA = R - FP/(FP+TN)



        mem_inf_acc = 0.5*(t_tr_acc + 1 - t_te_acc)
        print('For membership inference attack via correctness, the attack acc is {acc1:.3f}, with train acc {acc2:.3f} and test acc {acc3:.3f}'.format(acc1=mem_inf_acc, acc2=t_tr_acc, acc3=t_te_acc) )
        print('R:{R}, P:{P}, F1:{F1},FPR={FPR},MA={MA}'.format(R=R, P=P, F1=F1,FPR = FPR,MA=MA))
        return


    def _mem_inf_thre(self, v_name, s_tr_values, s_te_values, t_tr_values, t_te_values):
        # perform membership inference attack by thresholding feature values: the feature can be prediction confidence,
        # (negative) prediction entropy, and (negative) modified entropy
        t_tr_mem, t_te_non_mem = 0, 0
        f_tr_mem, f_te_non_mem = 0, 0
        for num in range(self.num_classes):
            thre = self._thre_setting(s_tr_values[self.s_tr_labels==num], s_te_values[self.s_te_labels==num])
            t_tr_mem += np.sum(t_tr_values[self.t_tr_labels==num]>=thre)

            t_te_non_mem += np.sum(t_te_values[self.t_te_labels==num]<thre)
            #f_tr_mem += np.sum(t_tr_values[self.t_tr_labels == num] >= thre)
            #f_te_non_mem += np.sum(t_te_values[self.t_te_labels == num] < thre)

        #print(f'{len(self.t_tr_labels)=}')
        TP = t_tr_mem
        FP = t_te_non_mem
        FN = len(self.t_tr_labels) - TP
        TN = len(self.t_te_labels) - FP

        # TP = t_tr_mem
        # FP = len(self.t_te_labels) - t_te_non_mem
        # FN = len(self.t_tr_labels) - TP
        # TN = len(self.t_te_labels) - FP

        #print('TP:{TP}, FP:{FP}, FN:{FN},TN={TN}'.format(TP=TP, FP=FP, FN=FN, TN=TN))


        #print(len(self.t_te_labels))
        #print(TP/len(self.t_tr_labels))

        R = TP / (TP + FN)
        # R = TP / self.length
        #P = TP/(TP + FP)
        P = TP/len(self.t_tr_labels) / (TP/len(self.t_tr_labels) + FP/len(self.t_te_labels))
        if(P+R == 0):
            F1 = None
        else:
            F1 = 2 * P * R / (P + R)
        # FPR = FP*((TP+FN)/self.length) / self.length
        FPR = FP / len(self.t_te_labels)
        # FPR = FP / self.length
        MA = R - FP / (FP + TN)
        acc = (TP + TN) / (TP+TN+FP+FN)
        #
        # mem_inf_acc = 0.5*(t_tr_mem/(len(self.t_tr_labels)+0.0) + t_te_non_mem/(len(self.t_te_labels)+0.0))
        # print('For membership inference attack via {n}, the attack acc is {acc:.3f}'.format(n=v_name,acc=mem_inf_acc))
        # print('R:{R}, P:{P}, F1:{F1},FPR={FPR},MA={MA}'.format(R=R, P=P, F1=F1,FPR = FPR,MA=MA))

        # R = TP / (TP + FN)
        # P = TP / (TP + FP)
        # F1 = 2 * P * R / (P + R)
        # FPR = FP / (FP + TN)
        # MA = R - FP / (FP + TN)
        #print('R:{R}, P:{P}, F1:{F1},FPR={FPR},MA={MA}'.format(R=R, P=P, F1=F1, FPR=FPR, MA=MA))

        return TP, FP, FN, TN,R, P, F1,FPR,MA,acc



    # def _mem_inf_thre(self, v_name, s_tr_values, s_te_values, t_tr_values, t_te_values):
    #     # perform membership inference attack by thresholding feature values: the feature can be prediction confidence,
    #     # (negative) prediction entropy, and (negative) modified entropy
    #
    #     for thre in range(5, 11):
    #         t_tr_mem, t_te_non_mem = 0, 0
    #         f_tr_mem, f_te_non_mem = 0, 0
    #         print('thre:{}'.format(thre/10))
    #         for num in range(self.num_classes):
    #             thre = thre/10
    #
    #             t_tr_mem += np.sum(t_tr_values[self.t_tr_labels==num]>=thre)
    #             t_te_non_mem += np.sum(t_te_values[self.t_te_labels==num]<thre)
    #             #f_tr_mem += np.sum(t_tr_values[self.t_tr_labels == num] >= thre)
    #             #f_te_non_mem += np.sum(t_te_values[self.t_te_labels == num] < thre)
    #
    #
    #         # TP = t_tr_mem/len(self.t_tr_labels)
    #         TP = t_tr_mem
    #         print('TP{}'.format(TP))
    #
    #         # FP = t_te_non_mem/len(self.t_te_labels)
    #         FP = t_te_non_mem
    #         print('FP{}'.format(FP))
    #         FN = len(self.t_tr_labels) - TP
    #         TN = len(self.t_te_labels) - FP
    #
    #         R = TP / (TP + FN)
    #         P = TP / (TP + FP)
    #         F1 = 2 * P * R / (P + R)
    #         FPR = FP / (FP + TN)
    #         MA = R - FP / (FP + TN)
    #
    #         mem_inf_acc = 0.5*(t_tr_mem/(len(self.t_tr_labels)+0.0) + t_te_non_mem/(len(self.t_te_labels)+0.0))
    #         print('For membership inference attack via {n}, the attack acc is {acc:.3f}'.format(n=v_name,acc=mem_inf_acc))
    #         print('R:{R}, P:{P}, F1:{F1},FPR={FPR},MA={MA}'.format(R=R, P=P, F1=F1,FPR = FPR,MA=MA))
    #
    #     return
    

    
    def _mem_inf_benchmarks(self, all_methods=True, benchmark_methods=[]):
        # if (all_methods) or ('correctness' in benchmark_methods):
        #     self._mem_inf_via_corr()
        # if (all_methods) or ('confidence' in benchmark_methods):
        #     self._mem_inf_thre('confidence', self.s_tr_conf, self.s_te_conf, self.t_tr_conf, self.t_te_conf)
        # if (all_methods) or ('entropy' in benchmark_methods):
        #     self._mem_inf_thre('entropy', -self.s_tr_entr, -self.s_te_entr, -self.t_tr_entr, -self.t_te_entr)
        if (all_methods) or ('modified entropy' in benchmark_methods):
            return_output = self._mem_inf_thre('modified entropy', -self.s_tr_m_entr, -self.s_te_m_entr, -self.t_tr_m_entr, -self.t_te_m_entr)

        return return_output