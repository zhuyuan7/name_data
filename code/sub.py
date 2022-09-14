import numpy  as np
import datetime
import time
from arg import arg_parser 


argp = arg_parser()


def flat_accuracy(preds, labels):
    
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):

    # 반올림
    elapsed_rounded = int(round((elapsed)))
    
    # hh:mm:ss으로 형태 변경
    return str(datetime.timedelta(seconds=elapsed_rounded))


def convert_to_text(text):
    text['target_label'] = text['target_label'].replace({0:"Korean",1:"French",2:"Irish"
                                                ,3: "Chinese",4:"Portuguese",5:"Dutch"
                                                ,6:"Japanese",7:"Russian", 8:"Czech"
                                                ,9:"Arabic",10:"Italian",11:"Vietnamese"
                                                ,12:"Polish",13:"German",14:"Spanish"
                                                ,15:"Scottish",16:"English",17:"Greek"})

    text['predicted_label'] = text['predicted_label'].replace({0:"Korean",1:"French",2:"Irish"
                                                ,3: "Chinese",4:"Portuguese",5:"Dutch"
                                                ,6:"Japanese",7:"Russian", 8:"Czech"
                                                ,9:"Arabic",10:"Italian",11:"Vietnamese"
                                                ,12:"Polish",13:"German",14:"Spanish"
                                                ,15:"Scottish",16:"English",17:"Greek"})
    
    a = text.to_csv(argp.predict_txt, index =False)

    return a

