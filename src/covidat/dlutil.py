import os
import ssl
import typing
import urllib.response
from datetime import datetime
from email.message import Message
from email.utils import parsedate_to_datetime
from http import HTTPStatus
from http.client import HTTPMessage, parse_headers
from logging import getLogger
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .util import Openable

logger = getLogger(__name__)


def create_unsafe_ssl_ctx() -> ssl.SSLContext:
    result = ssl.create_default_context()
    result.set_ciphers("DEFAULT@SECLEVEL=1")
    return result


FROM_EMAIL = os.environ["COVAT_BOT_FROM_EMAIL"]  # See doc below
"""We enforce specifying a contact email here,
see <https://www.rfc-editor.org/rfc/rfc9110.html#name-from>:

> The "From" header field contains an Internet email address for a human user
> who controls the requesting user agent.
> [...]
>
> A robotic user agent SHOULD send a valid From header field so that the
> person responsible for running the robot can be contacted if problems occur
> on servers, such as if the robot is sending excessive, unwanted, or invalid
> requests.
>
> ... its value is expected to be visible to anyone receiving or observing the
> request and is often recorded within logfiles and error reports without any
> expectation of privacy.

It would be wise to create a separate mail address for this purpose (but going
to a closely monitored / your main inbox).
"""


def get_moddate(hdrs: Message) -> datetime | None:
    dt = hdrs.get("Last-Modified") or hdrs.get("Date")
    return parsedate_to_datetime(typing.cast(str, dt)) if dt else None


def dl_if_not_modified(
    url: str, lastheaders: Message | None, dry_run: bool = False
) -> tuple[bool, HTTPError | urllib.response.addinfourl]:
    reqheaders = {
        "From": FROM_EMAIL,
    }
    if lastheaders:
        etag = lastheaders.get("Etag")
        if etag:
            reqheaders["If-None-Match"] = etag
        mdate = lastheaders.get("Last-Modified")
        if mdate:
            reqheaders["If-Modified-Since"] = mdate
    req = Request(url, headers=reqheaders)
    if dry_run:
        return False, HTTPError(
            req.full_url,
            HTTPStatus.NOT_MODIFIED,
            "Not Modified (dry run)",
            lastheaders or Message(),
            None,
        )
    try:
        resp = urlopen(req, context=create_unsafe_ssl_ctx())
    except HTTPError as e:
        if e.status == HTTPStatus.NOT_MODIFIED:
            return False, e
        raise HTTPError(url, e.code, url + ": " + e.reason, e.headers, e.fp) from e
    return True, resp


def _read_header_file(hdrfilepath: Openable) -> HTTPMessage | None:
    try:
        hdrf = open(hdrfilepath, "rb")
    except FileNotFoundError:
        return None
    with hdrf:
        return parse_headers(hdrf)


def write_hdr_file(resp_headers: Message, ofilepath: Openable, allow_existing=True) -> None:
    with open(ofilepath, "wb" if allow_existing else "xb") as of:
        of.write(resp_headers.as_bytes())


def dl_with_header_cache(
    url: str, hdrfilepath: Openable, dry_run: bool = False
) -> tuple[bool, HTTPError | urllib.response.addinfourl, HTTPMessage | None]:
    hdrs = _read_header_file(hdrfilepath)
    if not hdrs:
        logger.info("Could not read headers at %s (URL: %s)", hdrfilepath, url)
    modified, resp = dl_if_not_modified(url, hdrs, dry_run=dry_run)
    return modified, resp, hdrs
