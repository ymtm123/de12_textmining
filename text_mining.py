import pandas as pd
import collections
import itertools
import os
import glob

from scipy import sparse
import math

import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
from wordcloud import WordCloud


def get_co_df(doc):
    """
    ginzaのdocを受け取って、1文ごとに共起語の組み合わせをカウントする
    """

    sentences = list(doc.sents)
    # 各文の2-gramの組み合わせ
    sentence_combinations = [list(itertools.combinations(sentence, 2)) for sentence in sentences]

    # listをflatにする
    tc = []
    for sentence in sentence_combinations:
        tc.extend(sentence)

    # (word, pos)の組み合わせで共起語をカウント
    tc_set = [((t[0].text, t[0].pos_), (t[1].text, t[1].pos_)) for t in tc]

    # 出現回数
    ct = collections.Counter(tc_set)
    # ct.most_common()[:10]

    # sparce matrix
    # {単語, インデックス}の辞書作成
    tc_set_0 = [(t[0].text, t[0].pos_) for t in tc]
    tc_set_1 = [(t[1].text, t[1].pos_) for t in tc]

    ct_0 = collections.Counter(tc_set_0)
    ct_1 = collections.Counter(tc_set_1)

    dict_index_ct_0 = collections.OrderedDict((key[0], i) for i, key in enumerate(ct_0.keys()))
    dict_index_ct_1 = collections.OrderedDict((key[0], i) for i, key in enumerate(ct_1.keys()))
    dict_index_ct = collections.OrderedDict((key[0], i) for i, key in enumerate(ct.keys()))
    # print(dict_index_ct_0)

    #  単語の組合せと出現回数のデータフレームを作る
    word_combines = []
    for key, value in ct.items():
        word_combines.append([key[0][0], key[1][1], value, key[0][1], key[1][1]])

    df = pd.DataFrame([{
        'word1': i[0][0][0], 'word2': i[0][1][0], 'count': i[1]
        , 'word1_pos': i[0][0][1], 'word2_pos': i[0][1][1]
    } for i in ct.most_common()])

    return df


def get_cmap(df: pd.DataFrame):
    """
    Args:
      df(dataframe): 'word1', 'word2', 'count', 'word1_pos', 'word2_pos'

    Returns:
      {'ADP': 1, ...}

    """
    # 単語のposを抽出 indexで結合
    df_word_pos = pd.merge(pd.melt(df, id_vars=[], value_vars=['word1', 'word2'], value_name='word')
                           , pd.melt(df, id_vars=[], value_vars=['word1_pos', 'word2_pos'], value_name='pos')
                           , right_index=True, left_index=True).drop_duplicates(subset=['word', 'pos'])[['word', 'pos']]

    # posごとに色を付けたい
    cmap = set(df_word_pos['pos'].tolist())
    cmap = {k: v for v, k in enumerate(cmap)}

    return df_word_pos, cmap


def get_co_word(df: pd.DataFrame, word: str):
    """
    Args:
        df(pd.DataFrame):

    Returns:
        df_ex_co_word: 関連する単語のみを抽出する

    """

    # 特定のwordのみ抽出
    df_word = pd.concat([df[df['word1'] == word], df[df['word2'] == word]])

    # 単語のposを抽出 indexで結合
    df_word_pos = pd.merge(pd.melt(df_word, id_vars=[], value_vars=['word1', 'word2'], value_name='word')
                           , pd.melt(df_word, id_vars=[], value_vars=['word1_pos', 'word2_pos'], value_name='pos')
                           , right_index=True, left_index=True).drop_duplicates(subset=['word', 'pos'])[['word', 'pos']]

    # 特定の単語と関連する単語群の繋がり関係のみ抽出
    # 関連ワードがword1 or word2にある行を抽出
    df_ex_co_word = df[df[['word1', 'word2']].isin(list(df_word_pos['word'])).any(axis=1)]

    return df_ex_co_word


def get_network(df, edge_threshold):
    """
    df
    'word1', 'word2', 'count', 'word1_pos', 'word2_pos'
    """

    df_net = df.copy()

    # networkの定義
    nodes = list(set(df_net['word1'].tolist() + df_net['word2'].tolist()))

    graph = nx.Graph()
    #  頂点の追加
    graph.add_nodes_from(nodes)

    #  辺の追加
    #  edge_thresholdで枝の重みの下限を定めている
    for i in range(len(df_net)):
        row = df_net.iloc[i]
        if row['count'] > edge_threshold:
            graph.add_edge(row['word1'], row['word2'], weight=row['count'])

    # 孤立したnodeを削除
    isolated = [n for n in graph.nodes if len([i for i in nx.all_neighbors(graph, n)]) == 0]
    graph.remove_nodes_from(isolated)

    return graph


def plot_draw_networkx(df, edge_threshold, k, word=None, figsize=(8, 8)):
    """
    wordを指定していれば、wordとそれにつながるnodeを描画する
    """
    G = get_network(df, edge_threshold)

    # 面積を変えるために追加（山崎）
    # ない場合はwordを指定しない場合にエラーが生じる
    df = df[df["count"] > edge_threshold]

    plt.figure(figsize=figsize)
    # k = node間反発係数 weightが太いほど近い
    pos = nx.spring_layout(G, k=k)
    pr = nx.pagerank(G)

    # posごとに色を付けたい
    df_word_pos, c = get_cmap(df)

    cname = ['aquamarine', 'navy', 'tomato', 'yellow', 'yellowgreen'
        , 'lightblue', 'limegreen', 'gold'
        , 'red', 'lightseagreen', 'lime', 'olive', 'gray'
        , 'purple', 'brown' 'pink', 'orange']

    # cnameで指定する。品詞と数値の対応から、nodeの単語の色が突合できる
    cmap_all = [cname[c.get(df_word_pos[df_word_pos['word'] == node]['pos'].values[0])] for node in G.nodes()]

    # 出力する単語とつながりのある単語のみ抽出、描画
    words = []
    if word is not None:
        df_word = pd.concat([df[df['word1'] == word], df[df['word2'] == word]])

        words = list(pd.merge(pd.melt(df_word, id_vars=[], value_vars=['word1', 'word2'], value_name='word')
                              , pd.melt(df_word, id_vars=[], value_vars=['word1_pos', 'word2_pos'], value_name='pos')
                              , right_index=True, left_index=True).drop_duplicates(subset=['word', 'pos'])[
                         ['word', 'pos']]['word'])
        edges = list(df_word[['word1', 'word2']].apply(tuple, axis=1))
        sizes = np.array([pr[node] for node in words])
    else:
        # 面積を変えるために追加（山崎）
        edges = list(df[['word1', 'word2']].apply(tuple, axis=1))
        sizes = np.array([pr[node] for node in G.nodes()])

    cmap = [cname[c.get(df_word_pos[df_word_pos['word'] == node]['pos'].values[0])] for node in words]

    nx.draw_networkx_nodes(G, pos
                           , node_color=cmap if word is not None else cmap_all
                           , cmap=plt.cm.Reds
                           , alpha=0.3
                           , node_size=sizes * 30000
                           , nodelist=words if word is not None else G.nodes()  # 描画するnode
                           )
    # 日本語ラベル
    labels = {}
    for w in words:
        labels[w] = w
    nx.draw_networkx_labels(G, pos
                            , labels=labels if word is not None else None
                            , font_family='IPAexGothic'
                            , font_weight="normal"
                            )

    # 隣あう単語同士のweight
    edge_width = [G[edge[0]][edge[1]]['weight'] * 0.5 for edge in edges]
    nx.draw_networkx_edges(G, pos
                           # , edgelist=edges if word is not None else G.edges()
                           , edgelist=edges if word is not None else edges  # 面積を変えるために追加（山崎）
                           , alpha=0.5
                           , edge_color="darkgrey"
                           , width=edge_width if word is not None else edge_width
                           )

    plt.axis('off')
    plt.show()


def get_text_from_dir(text_dir):
    text_files = glob.glob("{}/*.txt".format(text_dir))
    text_data = ""
    for text_file in text_files:
        f = open(text_file, 'r', encoding="utf-8")
        text_data += f.read()
        f.close()

    df = pd.DataFrame({'文書': [text_data]})
    return df


def show_conetwork(text_dir, sw_list, word=None, edge_threshold=3, k=0.2):
    """共起ネットワークを表示する

    Parameters
    ----------
    text_dir : str
       テキストファイルがあるディレクトリ（ファイルは文字コードがUTF-8してあること）
    sw_list : list
       表示させたくない単語のリスト
    word : str
       ネットワークの中心に置きたい単語
    edge_threshold : int
       枝の重みの下限値
    k : float
       大きくするほど満遍なく配置される

    Returns
    -------
    None

    Examples
    --------
    show_conetwork(text_dir='./my_dir',
                   edge_threshold=3,
                   k=0.5,
                   sw_list=['こと', 'とき', 'もの', 'ところ', 'いう', 'お', 'しれ', '方', '〓'])

    """
    df = get_text_from_dir(text_dir)

    # exec spacy
    nlp = spacy.load('ja_ginza')
    docs = [nlp(s) for s in df['文書']]

    stop_words = nlp.Defaults.stop_words
    for sw in sw_list:
        stop_words.add(sw)

    # 共起語関係取得
    df_word_count = pd.concat([get_co_df(d) for d in docs]).reset_index(drop=True)

    # 除外する品詞
    # ADJ: 形容詞, ADP: 設置詞, ADV: 副詞, AUX: 助動詞, CCONJ: 接続詞,
    # DET: 限定詞, INTJ: 間投詞, NOUN: 名詞, NUM: 数詞,
    # PART: 助詞, PRON: 代名詞, PROPN: 固有名詞, PUNCT: 句読点,
    # SCONJ: 連結詞, SYM: シンボル, VERB: 動詞, X: その他
    extract_pos = ['ADP', 'AUX', 'CCONJ', 'INTJ', 'PART', 'PRON', 'PUNCT', 'SCONJ', 'SYM', 'X', 'SPACE']

    # df_ex_word_count = df_word_count[(~df_word_count['word1_pos'].isin(extract_pos)) \
    #                                  & (~df_word_count['word2_pos'].isin(extract_pos))]

    df_ex_word_count = df_word_count[(~df_word_count['word1_pos'].isin(extract_pos))
                                     & (~df_word_count['word2_pos'].isin(extract_pos))
                                     & (~df_word_count['word1'].isin(stop_words))
                                     & (~df_word_count['word2'].isin(stop_words))]

    # 共起回数を全文書でまとめておく
    df_net = df_ex_word_count.groupby(['word1', 'word2', 'word1_pos', 'word2_pos']).sum() \
        ['count'].sort_values(ascending=False).reset_index()

    if word:
        plot_draw_networkx(df_net, word=word, edge_threshold=edge_threshold, k=k)
    else:
        plot_draw_networkx(df_net, edge_threshold, k=k)


def get_count_df(doc):
    sentences = list(doc.sents)

    # listをflatにする
    tc = []
    for sentence in sentences:
        tc.extend(sentence)

    # (word, pos)の組み合わせでカウント
    tc_set = [(t.text, t.pos_) for t in tc]

    # 出現回数
    ct = collections.Counter(tc_set)

    #  単語の組合せと出現回数のデータフレームを作る
    word_combines = []
    for key, value in ct.items():
        word_combines.append([key[0], key[1], value])

    df = pd.DataFrame([{
        'word': i[0][0], 'count': i[1], 'word_pos': i[0][1]
    } for i in ct.most_common()])

    return df


def show_word_rank(text_dir, n_top, sw_list, title="出現頻度", font_size=8, bottom_position=0.2):
    """頻出単語トップを表示する

    Parameters
    ----------
    text_dir : str
       テキストファイルがあるディレクトリ（ファイルは文字コードがUTF-8してあること）
    n_top : int
       表示させたい上位の数
    sw_list : list
       表示させたくない単語のリスト
    title : str
       グラフのタイトル
    font_size : int
       フォントサイズ
    bottom_position : float
       グラフの下の位置（大きいほど上の方に表示される）

    Returns
    -------
    None

    Examples
    --------
    text_mining.show_word_rank(text_dir='./text/teacher/国際文化交流学科',
                               n_top=50,
                               sw_list=['こと', 'とき', 'もの', 'ところ', 'いう', 'お', 'しれ', '方', '〓'],
                               font_size=6,
                               bottom_position=0.20)
    """
    df = get_text_from_dir(text_dir)

    # exec spacy
    nlp = spacy.load('ja_ginza')
    docs = [nlp(s) for s in df['文書']]

    stop_words = nlp.Defaults.stop_words
    for sw in sw_list:
        stop_words.add(sw)

    df_count = pd.concat([get_count_df(d) for d in docs])

    # 記号 助詞　接続詞を除外
    extract_pos = ['ADP', 'AUX', 'CCONJ', 'INTJ', 'NUM', 'PART', 'PRON', 'PUNCT', 'SCONJ', 'SYM', 'X', 'SPACE']

    df_ex_word_count = df_count[(~df_count['word_pos'].isin(extract_pos))
                                & (~df_count['word'].isin(stop_words))]

    plt.rcParams['font.family'] = 'IPAexGothic'  # 日本語表示に必要
    fig, ax = plt.subplots(figsize=(6, 3), dpi=180)
    cmap = cm.get_cmap('Set3')
    color = 2

    x = np.array(list(range(n_top)))
    y = df_ex_word_count['count'][:n_top]
    plt.bar(x, y, color=cmap(color))

    ax.set_xticks(x)
    ax.set_xticklabels(df_ex_word_count['word'][:n_top], rotation=270, ha="center")
    ax.set_ylabel("回数")
    ax.set_title(title)
    ax.tick_params(labelsize=font_size)
    fig.subplots_adjust(bottom=bottom_position)
    plt.show()


def show_wordcloud(text_dir, sw_list, font_path):
    """WordCloudを表示する

    Parameters
    ----------
    text_dir : str
       テキストファイルがあるディレクトリ（ファイルは文字コードがUTF-8してあること）
    sw_list : list
       表示させたくない単語のリスト
    font_path : str
       日本語表示できるフォントファイル

    Returns
    -------
    None

    Examples
    --------
    text_mining.show_wordcloud(text_dir='./text/teacher/国際文化交流学科',
                               sw_list=['こと', 'とき', 'もの', 'ところ', 'いう', 'お', 'しれ', '方', '〓'],
                               font_path="C:/Users/Yamazaki/AppData/Local/Microsoft/Windows/Fonts/ipaexg.ttf")

    """
    df = get_text_from_dir(text_dir)

    # exec spacy
    nlp = spacy.load('ja_ginza')
    # docs = [nlp(s) for s in df['文書']]
    doc = nlp(df["文書"][0])

    stop_words = nlp.Defaults.stop_words
    for sw in sw_list:
        stop_words.add(sw)

    x = []
    for sent in doc.sents:
        for token in sent:
            if token.pos_ in ["PROPN", "NOUN", "VERB"]:
                x.append(token.lemma_)
    x = [c for c in x if c not in stop_words]

    # リストをスペース区切りの分かち書きに
    text_list = " ".join(x)

    # WordCloud定義
    wordcloud = WordCloud(width=600, height=400,
                          font_path=font_path,
                          background_color="white", collocations=False)

    # テキストからワードクラウドを生成する。
    wc = wordcloud.generate(text_list)

    plt.figure(figsize=(15, 12))
    plt.imshow(wc)
    plt.axis("off")
    plt.show()
