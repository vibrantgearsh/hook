// usage:
//   BLOCK_ENGINE_URL=https://ny.mainnet.block-engine.jito.wtf \
//   KEYPAIR_PATH=~/.config/solana/id.json \
//   RPC_URL=https://api.mainnet-beta.solana.com \
//   cargo run --release -- --tip-lamports 20000 --bundle-count 5

use std::{
    path::PathBuf,
    str::FromStr,
    sync::{Arc, RwLock},
    time::{Duration, SystemTime},
};

use clap::Parser;
use futures_util::StreamExt;
use jito_protos::{
    auth::{
        auth_service_client::AuthServiceClient, GenerateAuthChallengeRequest, GenerateAuthTokensRequest,
        RefreshAccessTokenRequest, Role, Token,
    },
    bundle::{bundle_result::Result as BResult, Bundle, BundleResult},
    convert::proto_packet_from_versioned_tx,
    searcher::{
        searcher_service_client::SearcherServiceClient, GetTipAccountsRequest, SendBundleRequest,
        SubscribeBundleResultsRequest,
    },
};
use rand::seq::SliceRandom;
use solana_client::nonblocking::rpc_client::RpcClient;
use solana_sdk::{
    pubkey::Pubkey,
    signature::{read_keypair_file, Keypair, Signer},
    system_instruction,
    transaction::{Transaction, VersionedTransaction},
};
use thiserror::Error;
use tokio::time::{sleep, timeout};
use tonic::{
    codegen::InterceptedService,
    service::Interceptor,
    transport::{Channel, ClientTlsConfig, Endpoint},
    Request, Status, Streaming,
};

#[derive(Debug, Error)]
enum BotErr {
    #[error("transport: {0}")]
    Transport(#[from] tonic::transport::Error),
    #[error("grpc: {0}")]
    Grpc(#[from] Status),
    #[error("rpc: {0}")]
    Rpc(#[from] solana_client::client_error::ClientError),
    #[error("bad pubkey from block engine: {0}")]
    Pubkey(#[from] solana_sdk::pubkey::ParsePubkeyError),
    #[error("couldn't read keypair at {0}: {1}")]
    Keypair(String, String),
    #[error("auth response was missing a token")]
    NoToken,
}

type Res<T> = Result<T, BotErr>;

#[derive(Parser, Debug)]
struct Args {
    /// e.g. https://ny.mainnet.block-engine.jito.wtf
    #[arg(long, env)]
    block_engine_url: String,

    #[arg(long, env)]
    keypair_path: PathBuf,

    #[arg(long, env)]
    rpc_url: String,

    #[arg(long, env, default_value_t = 10_000)]
    tip_lamports: u64,

    /// number of bundles
    #[arg(long, env, default_value_t = 1)]
    bundle_count: u32,
}

#[derive(Default)]
struct Tally {
    sent: u32,
    landed: u32,
}

impl Tally {
    fn rate(&self) -> f64 {
        if self.sent == 0 {
            0.0
        } else {
            self.landed as f64 / self.sent as f64
        }
    }
}

/// slaps the bearer token on every outgoing searcher-service call
#[derive(Clone)]
struct BearerAuth {
    token: Arc<RwLock<String>>,
}

impl Interceptor for BearerAuth {
    fn call(&mut self, mut req: Request<()>) -> Result<Request<()>, Status> {
        let t = self.token.read().unwrap();
        if !t.is_empty() {
            req.metadata_mut()
                .insert("authorization", format!("Bearer {t}").parse().unwrap());
        }
        Ok(req)
    }
}

async fn connect(url: &str) -> Res<Channel> {
    let mut ep = Endpoint::from_shared(url.to_owned())?;
    if url.starts_with("https") {
        ep = ep.tls_config(ClientTlsConfig::new())?;
    }
    Ok(ep.connect().await?)
}

/// challenge -> sign -> exchange, per the block engine's auth.proto
async fn auth_once(auth: &mut AuthServiceClient<Channel>, kp: &Keypair) -> Res<(Token, Token)> {
    let challenge = auth
        .generate_auth_challenge(GenerateAuthChallengeRequest {
            role: Role::Searcher as i32,
            pubkey: kp.pubkey().to_bytes().to_vec(),
        })
        .await?
        .into_inner()
        .challenge;

    // block engine wants the challenge signed as "<pubkey>-<challenge>"
    let to_sign = format!("{}-{challenge}", kp.pubkey());
    let sig = kp.sign_message(to_sign.as_bytes());

    let toks = auth
        .generate_auth_tokens(GenerateAuthTokensRequest {
            challenge: to_sign,
            client_pubkey: kp.pubkey().to_bytes().to_vec(),
            signed_challenge: sig.as_ref().to_vec(),
        })
        .await?
        .into_inner();

    Ok((
        toks.access_token.ok_or(BotErr::NoToken)?,
        toks.refresh_token.ok_or(BotErr::NoToken)?,
    ))
}

// token refresh loop. TODO: backoff
fn spawn_refresh(
    mut auth: AuthServiceClient<Channel>,
    token: Arc<RwLock<String>>,
    mut refresh: Token,
    mut exp: prost_types::Timestamp,
    kp: Arc<Keypair>,
) {
    tokio::spawn(async move {
        loop {
            let ttl = SystemTime::try_from(exp.clone())
                .unwrap()
                .duration_since(SystemTime::now())
                .unwrap_or(Duration::from_secs(0));
            sleep(ttl.saturating_sub(Duration::from_secs(60)).max(Duration::from_secs(5))).await;

            match auth
                .refresh_access_token(RefreshAccessTokenRequest {
                    refresh_token: refresh.value.clone(),
                })
                .await
            {
                Ok(resp) => {
                    let at = resp.into_inner().access_token.unwrap();
                    exp = at.expires_at_utc.clone().unwrap();
                    *token.write().unwrap() = at.value;
                }
                Err(e) => {
                    eprintln!("refresh failed ({e}), re-running full auth");
                    if let Ok((at, rt)) = auth_once(&mut auth, &kp).await {
                        exp = at.expires_at_utc.clone().unwrap();
                        *token.write().unwrap() = at.value;
                        refresh = rt;
                    }
                }
            }
        }
    });
}

async fn rand_tip_account(sc: &mut SearcherServiceClient<InterceptedService<Channel, BearerAuth>>) -> Res<Pubkey> {
    let accts = sc
        .get_tip_accounts(GetTipAccountsRequest {})
        .await?
        .into_inner()
        .accounts;
    let picked = accts.choose(&mut rand::thread_rng()).unwrap();
    Ok(Pubkey::from_str(picked)?)
}

/// waits (up to `timeout`) for a bundle result matching our uuid and reports whether it landed
async fn wait_for_land(stream: &mut Streaming<BundleResult>, uuid: &str) -> bool {
    let fut = async {
        while let Some(msg) = stream.next().await {
            let Ok(r) = msg else { continue };
            if r.bundle_id != uuid {
                // TODO: other in-flight bundles' results also come through this stream, we
                // should index pending sends by uuid instead of blocking one at a time
                continue;
            }
            match r.result {
                Some(BResult::Accepted(_)) => continue,
                Some(BResult::Processed(_)) | Some(BResult::Finalized(_)) => return true,
                Some(BResult::Rejected(_)) | Some(BResult::Dropped(_)) | None => return false,
            }
        }
        false
    };
    timeout(Duration::from_secs(30), fut).await.unwrap_or(false)
}

#[tokio::main]
async fn main() -> Res<()> {
    let args = Args::parse();

    let kp = read_keypair_file(&args.keypair_path)
        .map_err(|e| BotErr::Keypair(args.keypair_path.display().to_string(), e.to_string()))?;
    let kp = Arc::new(kp);

    let auth_chan = connect(&args.block_engine_url).await?;
    let mut auth_client = AuthServiceClient::new(auth_chan);
    let (access, refresh) = auth_once(&mut auth_client, &kp).await?;

    let token = Arc::new(RwLock::new(access.value));
    let exp = access.expires_at_utc.clone().unwrap();
    spawn_refresh(auth_client, token.clone(), refresh, exp, kp.clone());

    let searcher_chan = connect(&args.block_engine_url).await?;
    let mut sc = SearcherServiceClient::with_interceptor(searcher_chan, BearerAuth { token });

    let mut results = sc
        .subscribe_bundle_results(SubscribeBundleResultsRequest {})
        .await?
        .into_inner();
    let rpc = RpcClient::new(args.rpc_url.clone());

    let mut tally = Tally::default();

    for n in 0..args.bundle_count {
        let ta = rand_tip_account(&mut sc).await?;
        let bh = rpc.get_latest_blockhash().await?;

        let ix_tip = system_instruction::transfer(&kp.pubkey(), &ta, args.tip_lamports);

        let tx = VersionedTransaction::from(Transaction::new_signed_with_payer(
            &[ix_tip],
            Some(&kp.pubkey()),
            &[kp.as_ref()],
            bh,
        ));

        let pkt = proto_packet_from_versioned_tx(&tx);
        let resp = sc
            .send_bundle(SendBundleRequest {
                bundle: Some(Bundle {
                    header: None,
                    packets: vec![pkt],
                }),
            })
            .await?;
        let uuid = resp.into_inner().uuid;
        tally.sent += 1;

        let landed = wait_for_land(&mut results, &uuid).await;
        if landed {
            tally.landed += 1;
        }
        println!("bundle {n} uuid={uuid} tip_acct={ta} landed={landed}");
    }

    println!("{}/{} landed -> {:.0}%", tally.landed, tally.sent, tally.rate() * 100.0);
    Ok(())
}
