"""HTTP client for communicating with the Munasib Khareed FastAPI backend.

Provides a robust, typed client wrapping ``httpx`` to interact with all API
endpoints: health, metadata, product listings, cached thumbnails, reviews, full
analysis, and alternatives. Implements automatic retries with exponential backoff
on GET requests for transient connection errors and 5xx server responses.

All response methods return typed Pydantic models imported from
:mod:`munasib.api.schemas`. Failed HTTP requests and network issues are
translated into specific, friendly exceptions carrying the API's correlation
``request_id`` when available.

Note:
    This module must remain decoupled from Streamlit; no ``st.*`` calls or
    session state access are permitted here.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any

import httpx

from munasib.api.schemas import (
    AlternativesResult,
    AnalysisResponse,
    CategoryMetaResponse,
    HealthResponse,
    ProductDetail,
    ProductPage,
    ReviewPage,
)

#: Default base URL pointing to local FastAPI server.
DEFAULT_API_URL = "http://127.0.0.1:8000"

#: Environment variable used to override backend base URL.
ENV_API_URL = "MUNASIB_API_URL"

#: Default request timeout in seconds.
DEFAULT_TIMEOUT_SECONDS = 10.0

#: Maximum retry attempts for transient GET failures.
DEFAULT_MAX_RETRIES = 2

#: Initial backoff delay factor in seconds.
DEFAULT_BACKOFF_FACTOR = 0.1

#: Correlation header name matching FastAPI middleware.
REQUEST_ID_HEADER = "X-Request-ID"


class ApiClientError(Exception):
    """Base exception for all API client errors.

    Attributes:
        message: Human-readable error description.
        status_code: HTTP status code if a response was received, else None.
        request_id: Correlation ID extracted from headers or error envelope.
        detail: Detailed error breakdown (such as validation errors) if returned.
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        request_id: str | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.request_id = request_id
        self.detail = detail

    def __str__(self) -> str:
        req_suffix = f" [request_id={self.request_id}]" if self.request_id else ""
        code_suffix = f" (status={self.status_code})" if self.status_code else ""
        return f"{self.message}{code_suffix}{req_suffix}"


class ApiUnavailable(ApiClientError):
    """Raised when the backend API is unreachable, timed out, or connection refused."""


class ApiNotFound(ApiClientError):
    """Raised when the requested resource (e.g., ASIN) does not exist (HTTP 404)."""


class ApiValidationError(ApiClientError):
    """Raised when input parameters fail server-side validation (HTTP 422)."""


class ApiServerError(ApiClientError):
    """Raised when the backend server encounters an unhandled internal failure (HTTP 5xx)."""


class ApiClient:
    """Synchronous HTTP client for the Munasib Khareed API.

    Handles connection pooling, request retries for idempotent GET requests,
    correlation ID extraction, and typed Pydantic serialization.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Initialize the API client.

        Args:
            base_url: Base URL of the API. If None, reads from $MUNASIB_API_URL or defaults.
            timeout: Request timeout in seconds.
            max_retries: Number of retries for idempotent GET requests.
            backoff_factor: Delay multiplier between retries in seconds.
            transport: Optional custom HTTP transport (e.g., MockTransport for tests).
        """
        raw_url = base_url or os.environ.get(ENV_API_URL, DEFAULT_API_URL)
        self.base_url = raw_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.backoff_factor = backoff_factor

        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            transport=transport,
        )

    def close(self) -> None:
        """Close the underlying HTTP client session."""
        self._client.close()

    def __enter__(self) -> ApiClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _extract_request_id(
        self, response: httpx.Response, payload: Mapping[str, Any] | None
    ) -> str | None:
        """Extract the correlation ID from response headers or JSON body.

        Args:
            response: HTTP response object.
            payload: Parsed JSON payload dictionary, if available.

        Returns:
            The correlation ID string, or None if not present.
        """
        header_id = response.headers.get(REQUEST_ID_HEADER)
        if header_id:
            return str(header_id)
        if payload and isinstance(payload, dict):
            body_id = payload.get("request_id")
            if isinstance(body_id, str):
                return body_id
        return None

    def _handle_error_response(self, response: httpx.Response) -> None:
        """Translate a non-2xx HTTP response into a typed ApiClientError.

        Args:
            response: Failed HTTP response.

        Raises:
            ApiNotFound: For HTTP 404 responses.
            ApiValidationError: For HTTP 422 responses.
            ApiServerError: For HTTP 5xx responses.
            ApiClientError: For any other unhandled HTTP errors.
        """
        detail: Mapping[str, Any] | None = None
        message: str = response.text

        try:
            payload = response.json()
            if isinstance(payload, dict):
                message = payload.get("message") or response.text
                raw_detail = payload.get("detail")
                if isinstance(raw_detail, dict):
                    detail = raw_detail
        except Exception:
            payload = None

        request_id = self._extract_request_id(response, payload)

        if response.status_code == 404:
            raise ApiNotFound(
                message=message,
                status_code=404,
                request_id=request_id,
                detail=detail,
            )
        if response.status_code == 422:
            raise ApiValidationError(
                message=message,
                status_code=422,
                request_id=request_id,
                detail=detail,
            )
        if 500 <= response.status_code < 600:
            raise ApiServerError(
                message=message,
                status_code=response.status_code,
                request_id=request_id,
                detail=detail,
            )
        raise ApiClientError(
            message=message,
            status_code=response.status_code,
            request_id=request_id,
            detail=detail,
        )

    def _request_with_retry(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Execute an HTTP request with exponential backoff retries for GET operations.

        Args:
            method: HTTP method (e.g., 'GET').
            path: Target endpoint path relative to base_url.
            params: Optional query parameters mapping.

        Returns:
            Successful HTTP response.

        Raises:
            ApiUnavailable: If network connection failed after all retries.
            ApiClientError: If the server returned an error status code.
        """
        is_get = method.upper() == "GET"
        attempts = (self.max_retries + 1) if is_get else 1

        last_network_error: Exception | None = None

        for attempt in range(attempts):
            try:
                response = self._client.request(
                    method=method,
                    url=path,
                    params=params,
                )
                if is_get and 500 <= response.status_code < 600 and attempt < attempts - 1:
                    time.sleep(self.backoff_factor * (2**attempt))
                    continue

                if not response.is_success:
                    self._handle_error_response(response)
                return response

            except (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.ReadTimeout,
                httpx.WriteTimeout,
                httpx.PoolTimeout,
                httpx.NetworkError,
            ) as exc:
                last_network_error = exc
                if is_get and attempt < attempts - 1:
                    time.sleep(self.backoff_factor * (2**attempt))
                    continue
                break

        err_msg = (
            f"Unable to connect to Munasib Khareed API at {self.base_url}{path}: "
            f"{last_network_error}"
        )
        raise ApiUnavailable(message=err_msg) from last_network_error

    def health(self) -> HealthResponse:
        """Check API liveness and database reachability.

        Returns:
            HealthResponse model describing backend and database status.
        """
        resp = self._request_with_retry("GET", "/health")
        return HealthResponse.model_validate(resp.json())

    def get_category_meta(self) -> CategoryMetaResponse:
        """Fetch metadata, counts, and split distributions for the served category.

        Returns:
            CategoryMetaResponse model.
        """
        resp = self._request_with_retry("GET", "/meta/category")
        return CategoryMetaResponse.model_validate(resp.json())

    def search_products(
        self,
        query: str = "",
        limit: int = 20,
        offset: int = 0,
        split: str | None = None,
    ) -> ProductPage:
        """Search products with pagination and optional split filter.

        Args:
            query: Case-insensitive search query matching title or brand.
            limit: Page size (1 to 100).
            offset: Number of items to skip.
            split: Optional dataset split filter ('train', 'val', or 'test').

        Returns:
            ProductPage containing matching products and total count.
        """
        params: dict[str, Any] = {
            "query": query,
            "limit": limit,
            "offset": offset,
        }
        if split is not None:
            params["split"] = split

        resp = self._request_with_retry("GET", "/products", params=params)
        return ProductPage.model_validate(resp.json())

    def get_product(self, asin: str) -> ProductDetail:
        """Retrieve full profile for a single product by ASIN.

        Args:
            asin: Amazon Standard Identification Number.

        Returns:
            ProductDetail model.
        """
        resp = self._request_with_retry("GET", f"/products/{asin}")
        return ProductDetail.model_validate(resp.json())

    def get_reviews(
        self,
        asin: str,
        limit: int = 20,
        offset: int = 0,
        label: str | None = None,
        verified: bool | None = None,
    ) -> ReviewPage:
        """Retrieve reviews for a single product, sorted newest first.

        Args:
            asin: Product ASIN.
            limit: Page size (1 to 100).
            offset: Number of items to skip.
            label: Optional weak-supervision sentiment filter ('negative', 'neutral', 'positive').
            verified: Optional verified-purchase boolean filter.

        Returns:
            ReviewPage model.
        """
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }
        if label is not None:
            params["label"] = label
        if verified is not None:
            params["verified"] = verified

        resp = self._request_with_retry("GET", f"/products/{asin}/reviews", params=params)
        return ReviewPage.model_validate(resp.json())

    def get_analysis(self, asin: str) -> AnalysisResponse:
        """Retrieve the composed decision-support analysis for a product.

        Args:
            asin: Product ASIN.

        Returns:
            AnalysisResponse model.
        """
        resp = self._request_with_retry("GET", f"/products/{asin}/analysis")
        return AnalysisResponse.model_validate(resp.json())

    def get_alternatives(self, asin: str) -> AlternativesResult:
        """Retrieve evidence-based alternatives for a product.

        Args:
            asin: Product ASIN.

        Returns:
            AlternativesResult model.
        """
        resp = self._request_with_retry("GET", f"/products/{asin}/alternatives")
        return AlternativesResult.model_validate(resp.json())

    def get_product_image(self, asin: str) -> bytes | None:
        """Retrieve the locally cached JPEG thumbnail of a product.

        Images are display-only and served from the API's local cache, so this
        never fetches anything from Amazon. An ASIN without a cached thumbnail is
        a normal case, not an error: the ``404`` is translated to ``None`` so the
        caller can render a placeholder. Every other failure (an unreachable
        backend, a 5xx, a timeout) maps to the usual typed exceptions.

        Args:
            asin: Product ASIN.

        Returns:
            The raw JPEG bytes, or None when no thumbnail is cached for this ASIN.

        Raises:
            ApiClientError: For any failure other than a missing thumbnail.
        """
        try:
            resp = self._request_with_retry("GET", f"/products/{asin}/image")
        except ApiNotFound:
            return None
        return resp.content


def get_api_client(base_url: str | None = None) -> ApiClient:
    """Factory function returning a configured ApiClient instance.

    Args:
        base_url: Optional base URL override.

    Returns:
        Configured ApiClient instance.
    """
    return ApiClient(base_url=base_url)
