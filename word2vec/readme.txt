word2vec ����һ�ַ�����Ϊ CBOW, �����ǽ���� BOW ��
���Ҹ��㷢�����Ҵ���õ� word vector ��
�Ҷ� BOW ������, word2vec �������������ʵֵ��, ���� [0.12, 0.15, -0.33, ...] ������
�Ҷ��Թ� 1 ��, Ŀǰ���� glove 300d �Ľ�����, fasttext ��֮
�������� wod2vec, ������ glove 100d
�Ӹ����㷨�Ĺ������µ�Ԥѵ���õ�
Ȼ����ȡ�����������Ҫ�ĵ���, ��һ��������ݼ��еĵ��ʲ�������, ��ע��һ��Ŷ
fasttext �� facebook �����һ��ѵ���������ķ���
�������֮ǰ�ķ���, word2vec �Ѿ��ܿ���, ���� fasttext ʹ�����Է�����, ����, ���Խ����������
glove ��˹̹��������ķ���
��ʱ��ָ����, ��ʱ�����ѵ���õ�����
���ҷ���������µĻ�����, �Ҽ��� path_similarity, �� wup_similarity, ��0.1������
����Բο�һ���ټ�������������, ���Ǽ��������������Ƚϻ�ʱ��
���ٸ���˵һ��Ŷ, word2vec ���� glove ���� fasttext, ����ϰ�õ������Ǿ���һЩ��������Ե�
�Ҳ���϶�֪�� king-man+woman=queen ������
����, ���Ĵ󵨵�����Щ������


test["len_s1"] = df_test.sent1.apply(lambda x: len(x))

test["len_s2"] = df_test.sent2.apply(lambda x: len(x))

test["diff_len"] = test.len_s1 - test.len_s2

test["len_char_s1"] = df_test.sent1.apply(lambda x: len("".join(set(x.replace(" ", "")))))

test["len_char_s2"] = df_test.sent2.apply(lambda x: len("".join(set(x.replace(" ", "")))))

test["len_word_s1"] = df_test.sent1.apply(lambda x: len(x.split()))

test["len_word_s2"] = df_test.sent2.apply(lambda x: len(x.split()))

test["common_words"] = df_test.apply(lambda x: len(set(x["sent1"].lower().split()).intersection(
set(x["sent2"].lower().split()))), axis=1)


ds_test = np.array([document_similarity(s1, s2) for s1, s2 in zip(df_test.sent1, df_test.sent2)], dtype="float32")


test["path_similarity"] = ds_test[:, 0]

test["wup_similarity"] = ds_test[:, 1]




test["fuzz_qratio"] = df_test.apply(lambda x: fuzz.QRatio(x["sent1"], x["sent2"]), axis=1)

test["fuzz_wratio"] = df_test.apply(lambda x: fuzz.WRatio(x["sent1"], x["sent2"]), axis=1)

test["fuzz_partial_ratio"] = df_test.apply(lambda x: fuzz.partial_ratio(x["sent1"], x["sent2"]), axis=1)


test["fuzz_partial_token_set_ratio"] = df_test.apply(lambda x: fuzz.partial_token_set_ratio(x["sent1"], x["sent2"]), axis=1)

test["fuzz_partial_token_sort_ratio"] = df_test.apply(lambda x: fuzz.partial_token_sort_ratio(x["sent1"], x["sent2"]), axis=1)

test["fuzz_token_set_ratio"] = df_test.apply(lambda x: fuzz.token_set_ratio(x["sent1"], x["sent2"]), axis=1)

test["fuzz_token_sort_ratio"] = df_test.apply(lambda x: fuzz.token_sort_ratio(x["sent1"], x["sent2"]), axis=1)


test["sv1"] = df_test.sent1.apply(sent2vec)

test["sv2"] = df_test.sent2.apply(sent2vec)



test["wmd"] = df_test.apply(lambda x: wmd(x.sent1, x.sent2), axis=1)


test["norm_wmd"] = df_test.apply(lambda x: norm_wmd(x.sent1, x.sent2), axis=1)

test["cosine"] = test.apply(lambda x: distance.cosine(x.sv1, x.sv2), axis=1)

test["manhattan"] = test.apply(lambda x: distance.cityblock(x.sv1, x.sv2), axis=1)

test["jaccard"] = test.apply(lambda x: distance.jaccard(x.sv1, x.sv2), axis=1)

test["canberra"] = test.apply(lambda x: distance.canberra(x.sv1, x.sv2), axis=1)

test["euclidean"] = test.apply(lambda x: distance.euclidean(x.sv1, x.sv2), axis=1)

test["minkowski"] = test.apply(lambda x: distance.minkowski(x.sv1, x.sv2), axis=1)

test["braycurtis"] = test.apply(lambda x: distance.braycurtis(x.sv1, x.sv2), axis=1)

test["doc2vec_sim"] = np.array([doc2vec.docvecs.similarity(i1, i2) for i1, i2 in zip(range(3000, 3750), range(3750, 4500))], dtype="float32")

test["cosine_tfidf"] = np.array([pairwise.cosine_distances(tfidf_matrix[i1], tfidf_matrix[i2])[0][0] for i1, i2 in zip(range(3000, 3750), range(3750, 4500))], dtype="float32")

test["manhattan_tfidf"] = np.array([pairwise.manhattan_distances(tfidf_matrix[i1], tfidf_matrix[i2])[0][0] for i1, i2 in zip(range(3000, 3750), range(3750, 4500))], dtype="float32")

test["euclidean_tfidf"] = np.array([pairwise.euclidean_distances(tfidf_matrix[i1], tfidf_matrix[i2])[0][0] for i1, i2 in zip(range(3000, 3750), range(3750, 4500))], dtype="float32")

test["cosine_lsa"] = np.array([distance.cosine(lsa_matrix[i1], lsa_matrix[i2]) for i1, i2 in zip(range(3000, 3750), range(3750, 4500))], dtype="float32")

test["manhattan_lsa"] = np.array([distance.cityblock(lsa_matrix[i1], lsa_matrix[i2]) for i1, i2 in zip(range(3000, 3750), range(3750, 4500))], dtype="float32")

test["jaccard_lsa"] = np.array([distance.jaccard(lsa_matrix[i1], lsa_matrix[i2]) for i1, i2 in zip(range(3000, 3750), range(3750, 4500))], dtype="float32")

test["canberra_lsa"] = np.array([distance.canberra(lsa_matrix[i1], lsa_matrix[i2]) for i1, i2 in zip(range(3000, 3750), range(3750, 4500))], dtype="float32")
test["euclidean_lsa"] = np.array([distance.euclidean(lsa_matrix[i1], lsa_matrix[i2]) for i1, i2 in zip(range(3000, 3750), range(3750, 4500))], dtype="float32")
test["minkowski_lsa"] = np.array([distance.minkowski(lsa_matrix[i1], lsa_matrix[i2]) for i1, i2 in zip(range(3000, 3750), range(3750, 4500))], dtype="float32")
test["braycurtis_lsa"] = np.array([distance.braycurtis(lsa_matrix[i1], lsa_matrix[i2]) for i1, i2 in zip(range(3000, 3750), range(3750, 4500))], dtype="float32")


test["skew_s1"] = test.sv1.apply(stats.skew)
test["skew_s2"] = test.sv2.apply(stats.skew)
test["kurtosis_s1"] = test.sv1.apply(stats.kurtosis)
test["kurtosis_s2"] = test.sv2.apply(stats.kurtosis)
