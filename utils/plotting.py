# pages/basic_analysis.py の修正例 - line 322周辺のエラー対処

# 修正前（エラーの原因）
"""
fig = plotter.create_scatter_plot(...)
fig.update_layout(title=f"{x_channel} vs {y_channel}")  # figがNoneの場合エラー
"""

# 修正後（推奨）
def create_and_display_plot(plotter, plot_type, x_channel, y_channel=None, **kwargs):
    """プロット作成と表示のヘルパー関数"""
    try:
        if plot_type == "scatter":
            fig = plotter.create_scatter_plot(x_channel, y_channel, **kwargs)
        elif plot_type == "density":
            fig = plotter.create_density_plot(x_channel, y_channel, **kwargs)
        elif plot_type == "histogram":
            fig = plotter.create_histogram(x_channel, **kwargs)
        else:
            st.error(f"Unknown plot type: {plot_type}")
            return
        
        # figは修正版plotting.pyでは常にFigureオブジェクトが返される
        # 追加のレイアウト調整が必要な場合のみ実行
        if plot_type == "scatter" and y_channel:
            fig.update_layout(
                title=f"{x_channel} vs {y_channel}",
                xaxis_title=x_channel,
                yaxis_title=y_channel
            )
        
        # プロット表示
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"プロット作成エラー: {str(e)}")
        st.info("データの読み込みや変換に問題がある可能性があります。")

# メイン関数内での使用例
def main():
    # ... FCSファイル読み込み処理 ...
    
    if 'data' in st.session_state and st.session_state.data is not None:
        # PlottingUtilsの初期化
        plotter = PlottingUtils(st.session_state.data, st.session_state.metadata)
        
        # プロット設定UI
        with st.container():
            col1, col2 = st.columns(2)
            
            with col1:
                x_channel = st.selectbox("X軸チャンネル", options=list(st.session_state.data.columns))
                x_transform = st.selectbox("X軸変換", ["linear", "log10", "asinh", "biexponential"])
            
            with col2:
                y_channel = st.selectbox("Y軸チャンネル", options=list(st.session_state.data.columns))
                y_transform = st.selectbox("Y軸変換", ["linear", "log10", "asinh", "biexponential"])
        
        plot_type = st.radio("プロットタイプ", ["scatter", "density", "histogram"])
        
        # プロット作成と表示
        st.subheader("📊 データ可視化")
        
        if plot_type in ["scatter", "density"]:
            create_and_display_plot(
                plotter, plot_type, x_channel, y_channel,
                x_transform=x_transform, y_transform=y_transform
            )
        else:  # histogram
            create_and_display_plot(
                plotter, plot_type, x_channel,
                transform=x_transform
            )
        
        # 統計情報表示
        st.subheader("📈 統計情報")
        with st.expander("詳細統計", expanded=True):
            stats_df = plotter.create_statistics_table([x_channel, y_channel])
            if stats_df is not None:
                st.dataframe(stats_df)
            else:
                st.info("統計情報を計算できませんでした。")
    
    else:
        st.info("📁 FCSファイルをアップロードしてください。")

# データ読み込み関数の改良例
def load_fcs_file(uploaded_file):
    """FCSファイルの読み込み（エラーハンドリング強化版）"""
    try:
        # ここでFCS読み込み処理を実行
        # flowio, flowkit, fcsparserの順で試行
        
        data = None
        metadata = None
        
        # flowioを試行
        try:
            import flowio
            flow = flowio.FlowData(uploaded_file.getvalue())
            data = pd.DataFrame(flow.events, columns=flow.channels['$PnN'])
            metadata = flow.text
            library_used = "flowio"
        except Exception as e:
            st.warning(f"flowio読み込み失敗: {str(e)}")
            
            # flowkitを試行
            try:
                import flowkit
                sample = flowkit.Sample(uploaded_file.getvalue())
                data = sample.as_dataframe()
                metadata = sample.metadata
                library_used = "flowkit"
            except Exception as e:
                st.warning(f"flowkit読み込み失敗: {str(e)}")
                
                # fcsparserを試行
                try:
                    import fcsparser
                    metadata, data = fcsparser.parse(uploaded_file.getvalue())
                    library_used = "fcsparser"
                except Exception as e:
                    st.error(f"全ライブラリでの読み込みに失敗: {str(e)}")
                    return None, None, None
        
        if data is not None and len(data) > 0:
            st.success(f"✅ ファイル読み込み成功 ({library_used})")
            st.info(f"イベント数: {len(data):,}, パラメータ数: {len(data.columns)}")
            return data, metadata, library_used
        else:
            st.error("データが空または読み込みに失敗しました。")
            return None, None, None
            
    except Exception as e:
        st.error(f"ファイル読み込みエラー: {str(e)}")
        return None, None, None

if __name__ == "__main__":
    main()
