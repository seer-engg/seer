import pytest

from seer.api.subscriptions import stripe_service


def _build_fetcher(total: int):
    ids = [f"item_{idx}" for idx in range(total)]

    def _fetch(limit: int, starting_after: str | None = None, **_: object):
        start_idx = 0
        if starting_after:
            try:
                start_idx = ids.index(starting_after) + 1
            except ValueError:
                start_idx = len(ids)

        slice_ids = ids[start_idx:start_idx + limit]
        has_more = start_idx + limit < len(ids)
        return {
            "data": [{"id": item_id} for item_id in slice_ids],
            "has_more": has_more,
        }

    return _fetch


def test_paginate_stripe_list_handles_multiple_pages():
    fetcher = _build_fetcher(total=7)

    page1, has_more1 = stripe_service._paginate_stripe_list(fetcher, page=1, page_size=3)
    assert [item["id"] for item in page1] == ["item_0", "item_1", "item_2"]
    assert has_more1 is True

    page2, has_more2 = stripe_service._paginate_stripe_list(fetcher, page=2, page_size=3)
    assert [item["id"] for item in page2] == ["item_3", "item_4", "item_5"]
    assert has_more2 is True

    page3, has_more3 = stripe_service._paginate_stripe_list(fetcher, page=3, page_size=3)
    assert [item["id"] for item in page3] == ["item_6"]
    assert has_more3 is False


def test_paginate_stripe_list_returns_empty_when_offset_exceeds_items():
    fetcher = _build_fetcher(total=2)
    page, has_more = stripe_service._paginate_stripe_list(fetcher, page=3, page_size=2)
    assert page == []
    assert has_more is False


def test_paginate_stripe_list_validates_inputs():
    fetcher = _build_fetcher(total=1)
    with pytest.raises(ValueError):
        stripe_service._paginate_stripe_list(fetcher, page=0, page_size=10)
    with pytest.raises(ValueError):
        stripe_service._paginate_stripe_list(fetcher, page=1, page_size=0)
    with pytest.raises(ValueError):
        stripe_service._paginate_stripe_list(fetcher, page=1, page_size=101)
