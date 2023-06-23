model3 = readtable('model.csv');
[importance, scores]= fscchi2(model3, model3(:,1));
names = cell2table(model3.Properties.VariableNames);
names(:,1) = [];
names = table2array(names);

[~, sort_idx] = sort(scores);
names_feat_imp_desc_model3 = fliplr(names(sort_idx));
data_feat_imp_desc_model3 = fliplr(scores(sort_idx));