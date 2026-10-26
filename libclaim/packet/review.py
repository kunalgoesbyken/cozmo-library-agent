from libclaim.packet.models import ClaimPacket, ReviewEntry, Severity


def build_review_queue(packet: ClaimPacket) -> list[ReviewEntry]:
    queue: list[ReviewEntry] = []

    def add(ref_id: str, severity: Severity, reason: str, line_ref: str, frame_ref: str) -> None:
        queue.append(
            ReviewEntry(
                ref_id=ref_id,
                severity=severity,
                reason=reason,
                line_ref=line_ref,
                frame_ref=frame_ref,
            )
        )

    for index, book in enumerate(packet.books):
        line = f"books[{index}]"
        if book.status == "needs_appraisal":
            add(book.id, "high", "needs appraisal: not priced automatically", line, book.frame_ref)
        elif book.status == "unidentified":
            add(book.id, "medium", "spine could not be identified", line, book.frame_ref)
        elif book.replacement_cost.amount is None:
            add(book.id, "medium", "identified but no sourced replacement price", line, book.frame_ref)
        if not book.frame_ref:
            add(book.id, "low", "no frame reference", line, "")
    for index, item in enumerate(packet.items):
        line = f"items[{index}]"
        cost = item.replacement_cost
        if item.status == "needs_appraisal":
            add(item.id, "high", "needs appraisal: not priced automatically", line, item.frame_ref)
        elif cost.low is None or cost.high is None:
            add(item.id, "medium", "no sourced replacement price", line, item.frame_ref)
        if not item.frame_ref:
            add(item.id, "low", "no frame reference", line, "")
    if not packet.room.scale_method:
        add("room", "medium", "room dimensions have no scale method", "room", "")
    return queue
