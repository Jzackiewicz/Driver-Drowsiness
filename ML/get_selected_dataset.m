selected_feat_model3 = model3;                     

names_to_remove = names_feat_imp_desc_model3(1001:end);

idx_to_remove = ismember(selected_feat_model3.Properties.VariableNames, names_to_remove);

selected_feat_model3(:, idx_to_remove) = [];