# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 09:27:57 2017

@author: ozawa
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_wave(time, amplitude, xtitle = 'Time (s)', ytitle = 'Amplitude (arb.)', \
              hold = False, color = 'blue', marker = ',', legend = '', linestyle = '-', stem = False):
    '''  時間軸データと波形データを受け取り，波形を描く関数の定義
        引数 time:      時刻の離散データ
            amplitude: 上記の時刻データに対応する瞬時振幅値
            xtitle:    横軸のラベル（暗黙値は 'Time (s)')
            ytitle:    縦軸のラベル（暗黙値は 'Amplitude (arb.)'）
            hold:      False（暗黙値）のときは描画する。True のときは描画せず，データを保持する。
            color:     グラフの色（暗黙値はブルー）
            marker:    マーカの種類（暗黙値は pixel）
            legend:    凡例の文字列（暗黙値は「なし」）
            linestyle: 線の種類（暗黙値は実線）
            stem:    False（暗黙値）のときは折れ線グラフ。True のときは○付き棒グラフ。
    '''
    if (len(time) == 0):
        time = range(len(amplitude))
        if (xtitle == 'Time (s)'):
            xtitle = 'Time (point)'
    
    if (marker != ','):
        linestyle = ''

    if (stem == False):
        plt.plot(time, amplitude, color = color, marker = marker, linestyle = linestyle, label = legend)
    else:
        plt.stem(time, amplitude, label = legend, use_line_collection = True)

    if (legend != ''):
        plt.legend()
    
    if (hold == False):
        plt.xlabel(xtitle)
        plt.ylabel(ytitle)
        plt.show()

#============================
_called_first = True # これは，subplot を2回呼ばないようにするための逃げ口

def draw_FFT_spectrum(sp, fs = 48000.0, level = False, draw_range = 60.0, real_wave = False, \
                      phase_spectrum = True, stem = True, color = 'b', hold = False):
    ''' スペクトルを描く関数の定義 
          引数  sp:    FFTの結果として得られるスペクトル
                fs:    標本化周波数（暗黙値は 48 kHz）
                level: 縦軸を相対レベル表示とする場合に True
                draw_range: 相対レベル表示する場合における描画の縦軸範囲（暗黙値は 60 dB）
                real_wave:  実数の波形であることが既知の場合は，横軸はナイキスト周波数までに限定して描く
                phase_spectrum: True （暗黙値）では，位相スペクトルを描く。False では，振幅スペクトルのみ描く
                stem:  False のときは折れ線グラフ。True（暗黙値）のときは○付き棒グラフ。
                color: 色指定（暗黙値は 'b'ブルー）。
                hold:  False（暗黙値）のときは描画する。True のときは描画せず，データを保持する。
    '''
    
    global _called_first, _ax1_for_FFTspectrum
    
    amp = np.abs(sp)              # 振幅を求めます。
    phi = np.angle(sp)            # 位相を求めます。
        
    threshold = 10**(-draw_range / 20.0) * max(amp) # 振幅が描画範囲より小さい場合は，
    phi[ amp < threshold ] = 0.0                    # 位相は 0 と表示することにします。
    
    if (fs == None): # 標本化周波数がセットされていない場合には，横軸は正規化角周波数
        if (real_wave == False):
            omega_max = 2.0 * np.pi       # 正規化角周波数は 2πまで描画します。
            f_number = len(amp)           # 周波数ビンの数
        else:
            omega_max = np.pi             # 正規化角周波数は πまで描画します。
            f_number = int(len(amp) / 2)  # ナイキスト周波数までの周波数ビンの数
        frequency = np.linspace(0, omega_max, f_number, endpoint= False )
        xtitle = 'Normarized angular frequency (rad)'
    else:
        if (real_wave == False):
            f_max = fs                    # 標本化周波数まで描画します。
            f_number = len(amp)           # 周波数ビンの数
        else:
            f_max = fs / 2.                # ナイキスト周波数まで描画します。
            f_number = int(len(amp) / 2.)  # ナイキスト周波数までの周波数ビンの数
        frequency = np.linspace(0, f_max, f_number, endpoint= False) # 基本周波数とその整数倍の周波数を配列に納める。
        xtitle = 'Frequency (Hz)'
        
    ''' 振幅スペクトル '''
    ytitle = 'Amplitude (arb.)'
    
    if (level == True):                                                        # 縦軸が相対レベル (dB) のグラフを描く場合は，
        amp = amp + np.finfo(np.float64).eps                                   # 振幅が 0 の場合に log の計算でエラーを出さないよう計算機イプシロンを加えます。
        amp = 20.0 * np.log10(np.abs(amp[0:f_number]/np.max(amp[0:f_number]))) # 最大振幅を 0 dB とした相対レベルを計算します。
        # 描画の範囲を draw_range という引数で与えてあり，デフォルト値を 60 dB （マスキングを考慮した聴覚のダイナミックレンジ程度）までとします
        amp = amp + draw_range       # 関数 stem による描画の都合上，相対レベルに その数値を加えておきます。
        amp[amp < 0] = 0.0           # 描画の都合上，draw_range 未満のレベルは draw_range の下限に設定しておきます。
        ytitle = 'Relative level (dB)'
    
    # 重ね描きのため同じ subplot 領域を使うとエラーが生ずるので，１回のみしか宣言できない。
    if (_called_first == True):
        _ax1_for_FFTspectrum = plt.subplot(2, 1, 1)
        _called_first = False
    
    if (stem == True):
        _ax1_for_FFTspectrum.stem(frequency[0:f_number], amp[0: f_number])
    else:
        _ax1_for_FFTspectrum.plot(frequency[0:f_number], amp[0: f_number], color = color)
    if (xtitle == 'Normarized angular frequency (rad)'):
        if (real_wave == False):
            plt.xticks([0, np.pi/2.0, np.pi, 3.0*np.pi/2.0, 2.0*np.pi], ["$0$", "$0.5 \pi$" ,"$\pi$", "$1.5 \pi$", "$2 \pi$"])
        else:
            plt.xticks([0, np.pi/4.0, np.pi/2.0, 3.0*np.pi/4.0, np.pi], ["$0$", "$0.25 \pi$" ,"$0.5 \pi$", "$0.75 \pi$", "$\pi$"])
          
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    
    if (level == True):   # 縦軸が相対レベル (dB) のグラフを描く場合，先ほど draw_range の値を加えておいた分を差し引いた数値を表示します。
        plt.yticks([i for i in range(0, int(draw_range) + 1, 10)], [str(i) for i in range(-int(draw_range), 0+1, 10)])        
        
    if (phase_spectrum == False): # 位相スペクトルを描かなくてよいのであれば，ここで終了処理を行います。
        if (hold == False):
            plt.show()
            _called_first = True
        return
        
    ''' 位相スペクトル '''
    ytitle = 'Phase (rad)'
    plt.subplot(2, 1, 2)
    if (stem == True):
        plt.stem(frequency[0:f_number], phi[0: f_number], use_line_collection=True)
    else:
        plt.plot(frequency[0:f_number], phi[0: f_number])

    if (xtitle == 'Normarized angular frequency (rad)'):
        if (real_wave == False):
            plt.xticks([0, np.pi/2.0, np.pi, 3.0*np.pi/2.0, 2.0*np.pi], ["$0$", "$0.5 \pi$" ,"$\pi$", "$1.5 \pi$", "$2 \pi$"])
        else:
            plt.xticks([0, np.pi/4.0, np.pi/2.0, 3.0*np.pi/4.0, np.pi], ["$0$", "$0.25 \pi$" ,"$0.5 \pi$", "$0.75 \pi$", "$\pi$"])

    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.ylim(-np.pi, np.pi)
    plt.yticks([-np.pi, -np.pi/2.0, 0, np.pi/2.0, np.pi], ["$-\pi$", "$-\pi/2$" ,"0", "$\pi/2$", "$\pi$"])

    if (hold == False):
        plt.show()
        _called_first = True

#============================
def hanning(N):
    ''' ハニング窓を計算する関数
    　　引数 N: 窓長
    '''
    n = np.arange(0, N)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * (n + 0.5) / (N)))

#============================
def overlap_add(soundA, soundB, overlap = None):
    '''2音を重畳加算する関数の定義
       引数： soundA: 元となる音の配列
              soundB: soundA の最後の部分に soundB の一部を重ねる形で連結する
              shift:  重畳させる点数
    '''
    if (overlap == None):                # もし重畳させる点数が指定されていないのであれば
        overlap = int(len(soundB) / 2)   # 音B の半分の長さを重畳させる。
    elif (overlap == 0):
        return(np.concatenate([soundA, soundB]))
    
    if (len(soundA) < overlap):          # もし，soundA が「重畳するだけの長さがない」のであれば，
        soundA = np.concatenate([np.zeros(overlap-len(soundA)), soundA])    # soundB を崩さないよう soundA の先頭に 0 系列を埋め込んで重畳する。

    soundA = np.concatenate([soundA[0: -overlap], soundA[-overlap:] + soundB[0:overlap]])
    soundA = np.concatenate([soundA, soundB[overlap:]])
    return(soundA)

#============================
def circle_shift(wave, n_shift):
    ''' 時間波形 wave を 点数 n_shift だけ円状シフトする関数 '''
    return (np.r_[wave[-n_shift:], wave[:-n_shift]])

#============================
from pylab import *

def autocorr(x, nlags=None):
    """自己相関関数を求める
    x:     信号
    nlags: 自己相関関数のサイズ（lag=0からnlags-1まで）
           引数がなければ（lag=0からlen(x)-1まですべて）
    """
    N = len(x)
    if nlags == None: nlags = N
    r = np.zeros(nlags)
    for lag in range(nlags):
        for n in range(N - lag):
            r[lag] += x[n] * x[n + lag]
    return r

def LevinsonDurbin(r, lpcOrder):
    """Levinson-Durbinのアルゴリズム
    k次のLPC係数からk+1次のLPC係数を再帰的に計算して
    LPC係数を求める"""
    # LPC係数（再帰的に更新される）
    # a[0]は1で固定のためlpcOrder個の係数を得るためには+1が必要
    a = np.zeros(lpcOrder + 1)
    e = np.zeros(lpcOrder + 1)

    # k = 1の場合
    a[0] = 1.0
    a[1] = - r[1] / r[0]
    e[1] = r[0] + r[1] * a[1]
    lam = - r[1] / r[0]

    # kの場合からk+1の場合を再帰的に求める
    for k in range(1, lpcOrder):
        # lambdaを更新
        lam = 0.0
        for j in range(k + 1):
            lam -= a[j] * r[k + 1 - j]
        lam /= e[k]

        # aを更新
        # UとVからaを更新
        U = [1]
        U.extend([a[i] for i in range(1, k + 1)])
        U.append(0)

        V = [0]
        V.extend([a[i] for i in range(k, 0, -1)])
        V.append(1)

        a = np.array(U) + lam * np.array(V)

        # eを更新
        e[k + 1] = e[k] * (1.0 - lam * lam)

    return a, e[-1]
