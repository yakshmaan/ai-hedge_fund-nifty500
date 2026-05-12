"""
fetch_data_nifty500.py
----------------------
Downloads OHLCV data for all Nifty 500 stocks from Yahoo Finance.
Saves one CSV per stock into data/raw/.
 
Run this once to build your dataset. Re-run anytime to refresh.
 
Usage:
    python data/fetch_data_nifty500.py
"""
 
import yfinance as yf
import pandas as pd
import os
from datetime import datetime
 
# ── CONFIG ────────────────────────────────────────────────────────────────────
 
START_DATE = "2018-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")
RAW_DIR    = "data/raw"
 
# Nifty 500 symbols — all NSE listed with .NS suffix
NIFTY500_SYMBOLS = [
    # Nifty 50
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "BAJFINANCE.NS", "WIPRO.NS", "ONGC.NS", "NTPC.NS",
    "POWERGRID.NS", "ULTRACEMCO.NS", "BAJAJFINSV.NS", "HCLTECH.NS", "JSWSTEEL.NS",
    "TATASTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS", "DIVISLAB.NS",
    "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS", "HEROMOTOCO.NS", "HINDALCO.NS",
    "INDUSINDBK.NS", "M&M.NS", "NESTLEIND.NS", "SBILIFE.NS", "TATACONSUM.NS",
    "TECHM.NS", "CIPLA.NS", "APOLLOHOSP.NS", "BAJAJ-AUTO.NS", "BPCL.NS",
    "BRITANNIA.NS", "HDFCLIFE.NS", "UPL.NS", "TATAMOTORS.NS", "SHREECEM.NS",
    # Nifty Next 50
    "ADANIGREEN.NS", "ADANITRANS.NS", "AMBUJACEM.NS", "AUROPHARMA.NS", "BANDHANBNK.NS",
    "BANKBARODA.NS", "BERGEPAINT.NS", "BIOCON.NS", "BOSCHLTD.NS", "CHOLAFIN.NS",
    "COLPAL.NS", "CONCOR.NS", "DABUR.NS", "DLF.NS", "GAIL.NS",
    "GODREJCP.NS", "GODREJPROP.NS", "HAVELLS.NS", "ICICIGI.NS", "ICICIPRULI.NS",
    "INDUSTOWER.NS", "IRCTC.NS", "JUBLFOOD.NS", "LICI.NS", "LUPIN.NS",
    "MARICO.NS", "MCDOWELL-N.NS", "MUTHOOTFIN.NS", "NAUKRI.NS", "PAGEIND.NS",
    "PGHH.NS", "PIIND.NS", "PIDILITIND.NS", "PNBHOUSING.NS", "RECLTD.NS",
    "SAIL.NS", "SIEMENS.NS", "SRF.NS", "TORNTPHARM.NS", "TRENT.NS",
    "TVSMOTOR.NS", "UBL.NS", "VEDL.NS", "VOLTAS.NS", "WHIRLPOOL.NS",
    "YESBANK.NS", "ZEEL.NS", "ZOMATO.NS", "NYKAA.NS", "PAYTM.NS",
    # Nifty Midcap 150
    "ABCAPITAL.NS", "ABFRL.NS", "ABSLAMC.NS", "ACC.NS", "AARTIIND.NS",
    "AIAENG.NS", "AJANTPHARMA.NS", "ALKEM.NS", "APLLTD.NS", "APOLLOTYRE.NS",
    "ASHOKLEY.NS", "ASTRAL.NS", "ATUL.NS", "AUBANK.NS", "AWHCL.NS",
    "BAJAJHLDNG.NS", "BALKRISIND.NS", "BALRAMCHIN.NS", "BATAINDIA.NS", "BAYERCROP.NS",
    "BHARATFORG.NS", "BHEL.NS", "BIKAJI.NS", "BLUESTARCO.NS", "BRIGADE.NS",
    "CANFINHOME.NS", "CARBORUNIV.NS", "CASTROLIND.NS", "CEATLTD.NS", "CESC.NS",
    "CHAMBLFERT.NS", "CGPOWER.NS", "CHROMATIC.NS", "CLEAN.NS", "CMSINFO.NS",
    "COFORGE.NS", "CROMPTON.NS", "CUMMINSIND.NS", "CYIENT.NS", "DALBHARAT.NS",
    "DEEPAKNTR.NS", "DELTACORP.NS", "DMART.NS", "EIHOTEL.NS", "ELGIEQUIP.NS",
    "EMAMILTD.NS", "ENDURANCE.NS", "ESCORTS.NS", "EXIDEIND.NS", "ФАКТОРЫ.NS",
    "FIVESTAR.NS", "FLUOROCHEM.NS", "FORTIS.NS", "GLENMARK.NS", "GMRINFRA.NS",
    "GNFC.NS", "GODFRYPHLP.NS", "GRANULES.NS", "GRAPHITE.NS", "GSPL.NS",
    "HAPPSTMNDS.NS", "HFCL.NS", "HONAUT.NS", "IDFCFIRSTB.NS", "IEX.NS",
    "IGPL.NS", "INDHOTEL.NS", "INDIAMART.NS", "INDIANB.NS", "INDIGO.NS",
    "INTELLECT.NS", "IOC.NS", "IPCALAB.NS", "IRB.NS", "ISEC.NS",
    "JKCEMENT.NS", "JKTYRE.NS", "JMFINANCIL.NS", "JSL.NS", "JSWENERGY.NS",
    "KAJARIACER.NS", "KALPATPOWR.NS", "KANSAINER.NS", "KEC.NS", "KPIL.NS",
    "KPRMILL.NS", "KRBL.NS", "L&TFH.NS", "LALPATHLAB.NS", "LICHSGFIN.NS",
    "LINDEINDIA.NS", "LTIM.NS", "LTTS.NS", "MAHINDCIE.NS", "MANAPPURAM.NS",
    "MASFIN.NS", "MAXHEALTH.NS", "MCX.NS", "METROPOLIS.NS", "MFSL.NS",
    "MGL.NS", "MIDHANI.NS", "MINDTREE.NS", "MOTHERSON.NS", "MPHASIS.NS",
    "MRF.NS", "NATCOPHARM.NS", "NAVINFLUOR.NS", "NBCC.NS", "NCC.NS",
    "NILKAMAL.NS", "NLCINDIA.NS", "NMDC.NS", "NUVAMA.NS", "OBEROIRLTY.NS",
    "OFSS.NS", "OIL.NS", "OLDBRIDGE.NS", "ORIENTCEM.NS", "PATANJALI.NS",
    "PCBL.NS", "PERSISTENT.NS", "PETRONET.NS", "PFC.NS", "PFIZER.NS",
    "PHOENIXLTD.NS", "POLYCAB.NS", "POLYMED.NS", "POONAWALLA.NS", "PRESTIGE.NS",
    "PRINCEPIPE.NS", "PRSMJOHNSN.NS", "PSB.NS", "PVRINOX.NS", "RADICO.NS",
    "RAILTEL.NS", "RAIN.NS", "RAJESHEXPO.NS", "RAMCOCEM.NS", "RATNAMANI.NS",
    "RAYMOND.NS", "RITES.NS", "RVNL.NS", "SAFARI.NS", "SCHAEFFLER.NS",
    "SHYAMMETL.NS", "SJVN.NS", "SKFINDIA.NS", "SOBHA.NS", "SONACOMS.NS",
    "STARHEALTH.NS", "STYRENIX.NS", "SUBROS.NS", "SUMICHEM.NS", "SUNDARMFIN.NS",
    "SUNDRMFAST.NS", "SUNTV.NS", "SUPPETRO.NS", "SUPREMEIND.NS", "SUZLON.NS",
    "SYNGENE.NS", "TANLA.NS", "TATACOMM.NS", "TATAELXSI.NS", "TATAINVEST.NS",
    "TATAPOWER.NS", "TCNSBRANDS.NS", "TEAMLEASE.NS", "THERMAX.NS", "TIMKEN.NS",
    "TTKPRESTIG.NS", "TV18BRDCST.NS", "TVSHLTD.NS", "UCOBANK.NS", "UJJIVANSFB.NS",
    "UNITDSPR.NS", "UNIPARTS.NS", "UTIAMC.NS", "VAIBHAVGBL.NS", "VBL.NS",
    "VINATIORGA.NS", "VIPIND.NS", "WELCORP.NS", "WELSPUNLIV.NS", "WESTLIFE.NS",
    "WIPRO.NS", "WOCKPHARMA.NS", "ZENSARTECH.NS", "ZYDUSLIFE.NS",
    # Nifty Smallcap 250 (selection of liquid ones)
    "AADHARHFC.NS", "AARTIDRUGS.NS", "ABSLBANETF.NS", "ACCELYA.NS", "ACMESOLAR.NS",
    "ADFFOODS.NS", "ADORWELD.NS", "ADVENZYMES.NS", "AEGISLOG.NS", "AEROFLEX.NS",
    "AFFLE.NS", "AGROPHOS.NS", "AIIL.NS", "AKZOINDIA.NS", "ALIVUS.NS",
    "ALLCARGO.NS", "ALPA.NS", "AMARAJABAT.NS", "AMBER.NS", "AMJLAND.NS",
    "ANANTRAJ.NS", "ANDHRSUGAR.NS", "ANGELONE.NS", "ANURAS.NS", "APARINDS.NS",
    "APLAPOLLO.NS", "APTECHT.NS", "ARMANFIN.NS", "ARROWGREEN.NS", "ARTHURHIL.NS",
    "ARVIND.NS", "ARVINDFASN.NS", "ASAHIINDIA.NS", "ASHIANA.NS", "ASIANENE.NS",
    "ATGL.NS", "ATLANTA.NS", "ATUL.NS", "AVANTIFEED.NS", "AVINASH.NS",
    "AXISCADES.NS", "AYMSYNTEX.NS", "BAFNAPH.NS", "BAJAJCON.NS", "BAJAJHIND.NS",
    "BALMLAWRIE.NS", "BANARISUG.NS", "BANCOINDIA.NS", "BANSALAGRO.NS", "BASF.NS",
    "BASML.NS", "BBL.NS", "BBTC.NS", "BCHEMENGG.NS", "BEML.NS",
    "BFUTILITIE.NS", "BHAGERIA.NS", "BHANDARI.NS", "BHARAT.NS", "BHARATRAS.NS",
    "BHARTIHEXA.NS", "BIMETAL.NS", "BINDALAGRO.NS", "BIRLATYRES.NS", "BLKASHYAP.NS",
    "BLUEDART.NS", "BMATRIX.NS", "BNRfiltechn.NS", "BOROLTD.NS", "BPCL.NS",
    "BRNL.NS", "BROOKS.NS", "BSE.NS", "BSELINFRA.NS", "BSOFT.NS",
    "BURNPUR.NS", "BUTTERFLY.NS", "BVCL.NS", "BYKE.NS", "CALCOM.NS",
    "CAPACITE.NS", "CAPTRUST.NS", "CAMLINFINE.NS", "CANFINHOME.NS", "CARERATING.NS",
    "CASTEXTECH.NS", "CAVALCADE.NS", "CCHHL.NS", "CDSL.NS", "CENTURYPLY.NS",
    "CENTURYTEX.NS", "CERA.NS", "CGCL.NS", "CHALET.NS", "CHEMCON.NS",
    "CHEVIOT.NS", "CHIL.NS", "CIGNITITEC.NS", "CINEVISTA.NS", "CLNINDIA.NS",
    "CMRSL.NS", "COCHINSHIP.NS", "COFFEEDAY.NS", "COMPUSOFT.NS", "CONFIPET.NS",
    "CONSOFINVT.NS", "CONTROLPR.NS", "COSMOFILM.NS", "CPSEETF.NS", "CRAFTSMAN.NS",
    "CREATIVE.NS", "CREDITACC.NS", "CRISIL.NS", "CROWN.NS", "CUBEXTUB.NS",
    "DECCANCE.NS", "DECKOFFICE.NS", "DELHIVERY.NS", "DELTA.NS", "DENORA.NS",
    "DEVYANI.NS", "DFMFOODS.NS", "DGCONTENT.NS", "DHANI.NS", "DHARMAJ.NS",
    "DHRUV.NS", "DIACABS.NS", "DIGISPICE.NS", "DISHTV.NS", "DLINKINDIA.NS",
    "DMCC.NS", "DOLLAR.NS", "DPWIRES.NS", "DPWORLD.NS", "DRREDDY.NS",
    "DUCON.NS", "DUROPLY.NS", "DYNAMATECH.NS", "ECLERX.NS", "EDELWEISS.NS",
    "EDUCOMP.NS", "EIDPARRY.NS", "ELECTCAST.NS", "ELECON.NS", "ELECTHERM.NS",
    "EMAMIREAEL.NS", "ENGINERSIN.NS", "EPIGRAL.NS", "EQUITASBNK.NS", "ERIS.NS",
    "ESABINDIA.NS", "ESAFSFB.NS", "ESTER.NS", "ETHOSLTD.NS", "EUROBOND.NS",
    "EVEREADY.NS", "EXICOM.NS", "FACT.NS", "FAZE3Q.NS", "FCSSOFT.NS",
    "FEDERALBNK.NS", "FIEMIND.NS", "FINPIPE.NS", "FLEXITUFF.NS", "FMGOETZE.NS",
    "FOCUS.NS", "FOODSIN.NS", "FORCEMOT.NS", "GABRIEL.NS", "GALAXYSURF.NS",
]
 
# ── MAIN ──────────────────────────────────────────────────────────────────────
 
def fetch_stock(symbol: str) -> pd.DataFrame | None:
    try:
        df = yf.download(symbol, start=START_DATE, end=END_DATE, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df["Symbol"] = symbol.replace(".NS", "").replace("&", "_")
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        return df
    except Exception as e:
        return None
 
 
def fetch_all():
    os.makedirs(RAW_DIR, exist_ok=True)
    success = 0; failed = []
 
    # Deduplicate symbols
    symbols = list(dict.fromkeys(NIFTY500_SYMBOLS))
    print(f"Fetching {len(symbols)} stocks from {START_DATE} to {END_DATE}...\n")
 
    for i, symbol in enumerate(symbols, 1):
        print(f"[{i:03d}/{len(symbols)}] {symbol}", end=" ... ")
        df = fetch_stock(symbol)
        if df is not None:
            filename = symbol.replace(".NS", "").replace("&", "_") + ".csv"
            df.to_csv(os.path.join(RAW_DIR, filename))
            print(f"saved ({len(df)} rows)")
            success += 1
        else:
            print("FAILED")
            failed.append(symbol)
 
    print(f"\n{'─'*50}")
    print(f"Done. {success}/{len(symbols)} stocks fetched.")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed)}")
 
 
if __name__ == "__main__":
    fetch_all()
 