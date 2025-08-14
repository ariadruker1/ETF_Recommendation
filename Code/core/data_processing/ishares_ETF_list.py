import yfinance as yf

import streamlit as st


@st.cache_data(ttl=86400, show_spinner=False)
def download_valid_data():
    """
    Downloads historical data for a predefined list of ETFs from Yahoo Finance.

    This function attempts to download data for all tickers in a hardcoded list.
    It then filters out any tickers for which valid 'Adj Close' data is not available.
    The function is decorated with Streamlit's `cache_data` to prevent
    re-downloading the data on every rerun of the application, improving performance.

    Returns:
        tuple: A tuple containing:
            - valid_tickers (list): A list of ticker symbols that have valid data.
            - filtered_data (pd.DataFrame): A DataFrame with a multi-level index,
              containing only the 'Adj Close' prices for the valid tickers.
    """
    
    etf_list = ["SVR.TO", "CGL.TO", "XMV.TO", "XMI.TO", "XML.TO", "XIN.TO", "XMS.TO", "XMY.TO", "XEM.TO", "XMM.TO", "XEC.TO", "XUS.TO", "XEF.TO", "XMH.TO", "XMC.TO", "XDIV.TO", "XMU.TO", "XQQ.TO", "XWD.TO", "XDUH.TO", "XDG.TO", "XSU.TO", "XDU.TO", "XSUS.TO", "XSEA.TO", "XDGH.TO", "XESG.TO", "XGI.TO", "XCD.TO", "XSEM.TO", "XSP.TO", "CWO.TO", "CRQ.TO", "XID.TO", "XCH.TO", "XEMC.TO", "XHC.TO", "XDRV.TO", "CWW.TO", "XCV.TO", "XCG.TO", "XUSR.TO", "XDV.TO", "XDSR.TO", "XEU.TO", "CEW.TO", "XEH.TO", "XUU.TO", "COW.TO", "CIF.TO", "CYH.TO", "XDNA.TO", "XCLN.TO", "XQQU.TO", "XEXP.TO", "XAW.TO", "XHAK.TO", "XETM.TO", "XCHP.TO", "CIE.TO", "XUSF.TO", "XAD.TO", "XEN.TO", "CUD.TO", "CDZ.TO", "XQLT.TO", "XIU.TO", "CJP.TO", "XEG.TO", "XST.TO", "XIC.TO", "CPD.TO", "XSMC.TO",
                "XMA.TO", "XUSC.TO", "XSMH.TO", "XFH.TO", "XIT.TO", "XFN.TO", "XMTM.TO", "XBM.TO", "XEI.TO", "XVLU.TO", "XMD.TO", "XUT.TO", "XCSR.TO", "XPF.TO", "XHU.TO", "XGD.TO", "XSPC.TO", "XUH.TO", "XCS.TO", "XHD.TO", "CLU.TO", "XMW.TO", "XSC.TO", "XSE.TO", "CMR.TO", "CLG.TO", "CBH.TO", "CLF.TO", "CBO.TO", "CVD.TO", "XQB.TO", "XAGG.TO", "XCBG.TO", "XSHG.TO", "XAGH.TO", "XSTB.TO", "XFLB.TO", "XFLI.TO", "XFLX.TO", "XSAB.TO", "XTLH.TO", "XTLT.TO", "XFR.TO", "XGB.TO", "XCB.TO", "XSB.TO", "XSI.TO", "XRB.TO", "XLB.TO", "XHB.TO", "XBB.TO", "XSH.TO", "XSTH.TO", "XSTP.TO", "XCBU.TO", "XIGS.TO", "XSHU.TO", "XEB.TO", "XIG.TO", "XHY.TO", "GCNS.TO", "GGRO.TO", "GEQT.TO", "GBAL.TO", "XGRO.TO", "XBAL.TO", "FIE.TO", "XTR.TO", "XCNS.TO", "XEQT.TO", "XINC.TO", "CGR.TO", "XRE.TO"]

    data = yf.download(etf_list, period="max", group_by='ticker',
                       auto_adjust=False, progress=False)

    valid_tickers = []
    for ticker in etf_list:
        try:
            if (ticker, 'Adj Close') in data.columns and not data[(ticker, 'Adj Close')].dropna().empty:
                valid_tickers.append(ticker)
        except Exception:
            continue

    filtered_data = data.loc[:, [(ticker, 'Adj Close')
                                 for ticker in valid_tickers]]

    return valid_tickers, filtered_data