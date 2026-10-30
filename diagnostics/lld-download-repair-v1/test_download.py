import copy
import io
import json
import os
from pathlib import Path
import ssl
import tempfile
import unittest
from unittest.mock import patch,Mock
import urllib.error
import urllib.request
import download as N


class Tests(unittest.TestCase):
    def test_official_pin_no_revision_or_digest_bypass(self):
        with self.assertRaises(ValueError): N.verify_inventory({'repo':N.D.REPO,'revision':N.D.REVISION})
        value={'b':2,'a':1}
        self.assertEqual(N.digest(value),N.digest({'a':1,'b':2}))
        with patch.object(N,'CATALOG_SHA256',N.digest(value)):
            self.assertEqual(N.verify_inventory(value),value)
            with self.assertRaises(ValueError): N.verify_inventory({'a':2,'b':2})

    def test_direct_ignores_environment_proxy_without_global_config_edit(self):
        with patch.dict(os.environ,{'http_proxy':'http://bad.invalid:9','https_proxy':'http://bad.invalid:9'}), patch.object(N.urllib.request,'build_opener') as build:
            N.Transport(N.OFFICIAL,'direct')
            self.assertEqual(build.call_args.args[0].proxies,{})
            self.assertEqual(os.environ['https_proxy'],'http://bad.invalid:9')

    def test_explicit_proxy_is_validated_and_no_credentials(self):
        for url in ('http://127.0.0.1:1234','http://localhost:1234','socks5://example.org:5',
                    'http://name:password@example.org:8','http://example.org:8/path','http://example.org:8?token=secret'):
            with self.assertRaises(ValueError): N.Transport(N.OFFICIAL,'explicit',url)
        with patch.dict(os.environ,{'NO_PROXY':'*'}), patch.object(N.urllib.request,'build_opener') as build:
            N.Transport(N.OFFICIAL,'explicit','http://host.docker.internal:1234')
            self.assertEqual(build.call_args.args[0].proxies['https'],'http://host.docker.internal:1234')
            self.assertNotIn('NO_PROXY',os.environ)

    def test_endpoint_allowlist_and_proxy_mode(self):
        for endpoint in ('http://hf-mirror.com','https://evil.invalid','https://hf-mirror.com@evil.invalid'):
            with self.assertRaises(ValueError): N.Transport(endpoint)
        with self.assertRaises(ValueError): N.Transport(N.OFFICIAL,'bad')
        with self.assertRaises(ValueError): N.Transport(N.OFFICIAL,'direct','http://example.org')

    def test_url_rewrite_keeps_repo_revision_range_and_no_tokens(self):
        tr=N.Transport('https://hf-mirror.com','direct'); tr.opener=Mock()
        url=N.OFFICIAL+'/datasets/'+N.D.REPO+'/resolve/'+N.D.REVISION+'/images/a%2Bb.nii.gz?download=true'
        req=urllib.request.Request(url,headers={'Range':'bytes=4-','User-Agent':'test'})
        tr(req,timeout=17)
        actual=tr.opener.open.call_args.args[0]
        self.assertEqual(actual.full_url,url.replace(N.OFFICIAL,'https://hf-mirror.com'))
        self.assertEqual(actual.headers['Range'],'bytes=4-')
        self.assertEqual(tr.opener.open.call_args.kwargs['timeout'],17)
        with self.assertRaises(ValueError): tr(urllib.request.Request(url,headers={'Authorization':'Bearer secret'}))
        with self.assertRaises(ValueError): tr(urllib.request.Request(url,data=b'x'))
        with self.assertRaises(ValueError): tr('https://huggingface.co.evil.invalid/file')

    def test_error_does_not_leak_proxy_or_signed_urls(self):
        tr=N.Transport(N.OFFICIAL); tr.opener=Mock()
        tr.opener.open.side_effect=urllib.error.URLError(ConnectionRefusedError(111,'user:secret@proxy'))
        with self.assertRaises(ConnectionError) as caught: tr(N.OFFICIAL+'/api/datasets/test')
        self.assertIn('connection_refused',str(caught.exception))
        self.assertNotIn('secret',str(caught.exception))
        self.assertEqual(N.error_kind(urllib.error.URLError(ssl.SSLError('secret'))),'tls_verification_or_handshake')

    def test_metadata_retries_connection_failure_then_verifies(self):
        obj={'a':1}; tr=Mock(side_effect=[ConnectionError(),io.BytesIO(b'{}')])
        with patch.object(N.D,'catalog',return_value=obj),patch.object(N,'CATALOG_SHA256',N.digest(obj)),patch.object(N.time,'sleep'):
            self.assertEqual(N.metadata(tr),obj)
        self.assertEqual(tr.call_count,2)

    def test_metadata_integrity_failure_not_retried(self):
        tr=Mock(return_value=io.BytesIO(b'{}'))
        with patch.object(N.D,'catalog',return_value={'tampered':True}):
            with self.assertRaises(ValueError): N.metadata(tr)
        self.assertEqual(tr.call_count,1)

    def test_metadata_failure_bounded(self):
        tr=Mock(side_effect=ConnectionError())
        with patch.object(N.time,'sleep'):
            with self.assertRaises(ConnectionError): N.metadata(tr)
        self.assertEqual(tr.call_count,3)

    def test_preserves_existing_download_and_uses_explicit_transport(self):
        inv={'repo':N.D.REPO,'revision':N.D.REVISION,'files':{'README.md':{'size':3}},'scans':[]}
        with tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP')) as d:
            root=Path(d); N.D.save(root/'download-manifest.json',inv)
            with patch.object(N,'CATALOG_SHA256',N.digest(inv)),patch.object(N,'metadata') as meta,patch.object(N.D,'fetch_file') as fetch:
                tr=Mock(); N.download(root,True,tr)
                meta.assert_not_called()
                self.assertIs(fetch.call_args.kwargs['opener'],tr)
                self.assertEqual(N.D.read(root/'download-manifest.json'),inv)
                self.assertTrue(N.D.read(root/'download-complete.json')['complete'])

    def test_terms_before_network_and_mutation(self):
        with patch.object(N,'metadata') as meta,patch.object(N.D,'freeze') as freeze:
            with self.assertRaises(ValueError): N.download(Path('not-used'),False,Mock())
            meta.assert_not_called(); freeze.assert_not_called()

    def test_existing_catalog_tamper_stops_before_file_download(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP')) as d:
            root=Path(d); N.D.save(root/'download-manifest.json',{'tampered':True})
            with patch.object(N.D,'fetch_file') as fetch:
                with self.assertRaises(ValueError): N.download(root,True,Mock())
                fetch.assert_not_called()


if __name__=='__main__': unittest.main()
