import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.inspection import permutation_importance

def get_metric_by_sample_size(voc_type, voc_df, sample_sizes, features, seed, test_size, target ='species', n_trees = 500):
    """
    train machine learning models (random )
    
    """
    
    
    
    
    #get the vocalizations that belong to voc_type
    all_annotations = voc_df.loc[voc_df['human_label'] == voc_type]
        
    # train multiple models on different numbers of vocalizations
    all_scores = []
    all_importances = []
    
    #downsample
    all_downsampled = []
    for sample_n in sample_sizes:
        
        print('training', model_type, 'on', sample_n, 'vocalizations.')
        
        #sample equally for each species
        print('\tsampling data...')
        downsampled_list = []
        species_list = ['BW', 'BK', 'NB', 'SW', 'PO', 'LO', 'GO', 'LL']
        for species in species_list:

            #get the vocs for this species
            temp= all_annotations.loc[all_annotations['species'] == species]

            #sample
            temp = temp.sample(n=sample_n, random_state = seed)

            #update downsampled_list
            downsampled_list.append(temp)

        #assemble all the species
        ds_df = pd.concat(downsampled_list)
        ds_df = ds_df.reset_index(drop=True)
        
        #select the features
        ds_df = ds_df[features+[target]]

        #assert there are no missing data
        assert df.isnull().values.any() == False, "There are missing data"

        #split the train/test
        #choose the data and the label and convert to numpy array
        X = np.array(ds_df.drop(columns=[target]))
        y = np.array(ds_df[target])

        #split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)

        #train the model
        print('\ttraining model...')
        model = RandomForestClassifier(n_estimators = n_estimators,  
                                          random_state = random_state, 
                                          bootstrap = True,
                                          oob_score=True)
        model.fit(X_train, y_train)

        #get the clafficiation report
        print('\tevaluating model...')
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cr_df = pd.DataFrame(report).transpose()
        cr_df['model_type'] = model_type
        cr_df['sample_size'] = sample_n
        cr_df['voc_type'] = voc_type
        cr_df = cr_df.reset_index().rename(columns={'index':'species'})
        all_scores.append(cr_df)
         
    #combine the reports from all the sample sizes toegther
    all_scores = pd.concat(all_scores)
    all_importances = pd.concat(all_importances)
    print('done.')

    return all_scores