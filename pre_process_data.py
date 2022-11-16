import copy
import numpy as np
import pandas as pd




def main():
    columns = ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5', 'x6', 'y6', 'x7', 'y7', 'x8',
               'y8', 'x9', 'y9', 'x10', 'y10', 'x11', 'y11', 'x12', 'y12', 'x13', 'y13', 'x14', 'y14', 'x15', 'y15',
               'x16','y16', 'x17', 'y17', 'x18', 'y18', 'x19', 'y19', 'x20', 'y20']
    df = pd.read_csv('data/gestures.csv')
    df_labels = df['label']
    df_coordinates = df[columns]
    df_processed = pd.DataFrame(columns=columns)
    for index, row in df_coordinates.iterrows():
        temp_list = preprocess_landmark(row.tolist())
        df_processed = df_processed.append(pd.Series(temp_list, index=columns), ignore_index=True)

    df_processed['label'] = df_labels
    df_processed.to_csv('data/processed.csv', index=False)
    print(df.size)
    print(df_processed.size)
    print(df_processed.head())






def preprocess_landmark(landmark_list):
    temp_lm_list = copy.deepcopy(landmark_list)

    base_x=landmark_list[0]
    base_y=landmark_list[1]

    for i in range(len(landmark_list)):
        if(i%2==0):
            temp_lm_list[i] = temp_lm_list[i] - base_x
        else:
            temp_lm_list[i] = temp_lm_list[i] - base_y

    #temp_lm_list = list(itertools.chain.from_iterable(temp_lm_list))

    max_value = max(list(map(abs, temp_lm_list)))

    def normalize_(n):
        return n/max_value

    temp_lm_list = list(map(normalize_, temp_lm_list))

    return temp_lm_list


if __name__ == '__main__':
    main()